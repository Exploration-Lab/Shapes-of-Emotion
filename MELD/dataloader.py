import torch, pickle, pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class MELDCategorical(Dataset):    
    def __init__(self, path, bert_vectors, siamese_vectors, visual_vectors, train=True, n_classes = 7,  ):
        self.n_classes = n_classes
        self.videoIDs, self.videoSpeakers, self.labels_emotion, self.videoText,\
        self.videoAudio, self.videoSentence, self.trainVid,\
        self.testVid, self.labels_sentiment = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid) if x in self.videoIDs or x in self.testVid]


        self.bert_vectors = pickle.load(open(bert_vectors, 'rb'), encoding='latin1')
        self.siamese_vectors = pickle.load(open(siamese_vectors, 'rb'), encoding='latin1')
        self.videoLabels = self.labels_sentiment if self.n_classes == 3 else self.labels_emotion
        self.visual_vectors = pickle.load(open(visual_vectors, 'rb'), encoding='latin1')
        
        visual_vector_ids = [int(x.split('[')[0]) for x in self.visual_vectors]
        # print(visual_vector_ids)
        new_keys = []
        for x in self.keys:
            found = True
            for i in range(len(self.videoLabels[x])):
                if str(x)+'['+str(i)+']' not in self.visual_vectors:
                    found = False
                    break
            if found:
                new_keys.append(x)

        self.keys = new_keys

        self.len = len(self.keys)
        
    def get_linguistic_feautures_bert(self, vid):
        num_utterance = len(self.videoLabels[vid])
        features = []
        for j in range(num_utterance):
            bert_key = f"{vid}[{j}]"
            features.append(self.bert_vectors[bert_key])
        return np.array(features)
    def get_linguistic_feautures_sbert(self, vid):
        num_utterance = len(self.videoLabels[vid])
        features = []
        for j in range(num_utterance):
            bert_key = f"{vid}[{j}]"
            features.append(self.siamese_vectors[bert_key])
        return np.array(features)
    def get_visual_feautures(self, vid):
        num_utterance = len(self.videoLabels[vid])
        features = []
        for j in range(num_utterance):
            visual_key = f"{vid}[{j}]"
            features.append(self.visual_vectors[visual_key])
        return np.array(features)

    def get_labels(self, vid):
        num_utterance = len(self.videoLabels[vid])
        labels = []
        for j in range(num_utterance):
            label_key = f"{vid}[{j}]"
            labels.append(self.labels[label_key])
        return labels
    def __getitem__(self, index):
        vid = self.keys[index]
        # print(vid)
        # print(self.get_labels(vid))
        num_utterance = len(self.videoLabels[vid])
        bert_textf = torch.FloatTensor(self.get_linguistic_feautures_bert(vid))
        
        textf = torch.FloatTensor(self.get_linguistic_feautures_sbert(vid))
        acouf = torch.FloatTensor(self.videoAudio[vid])
        visualf = torch.FloatTensor(self.get_visual_feautures(vid))

        # print(textf.shape, acouf.shape, visualf.shape)
        prior_textf = torch.zeros(textf.size())
        prior_acouf = torch.zeros(acouf.size())
        prior_visualf = torch.zeros(visualf.size())

        prior_labels = []
        emotion_shift = []


        predecessor_label = 0
        # print(self.videoLabels[vid])
        for j in range(num_utterance):
            label_val = self.labels_sentiment[vid][j]
            if j!=0:
                prior_textf[j] = textf[j-1].clone()
                prior_acouf[j] = acouf[j-1].clone()
                prior_visualf[j] = visualf[j-1].clone()
                predecessor_label = self.labels_sentiment[vid][j-1]
                
            emotion_shift.append(1-((predecessor_label in [1] and label_val in [2]) or (label_val in [2] and predecessor_label in [1])))
            prior_labels.append(predecessor_label)
        
        # print(textf.shape, acouf.shape, visualf.shape)
        # print(prior_textf.shape, prior_acouf.shape, prior_visualf.shape)
        return bert_textf,\
               visualf,\
               acouf,\
               prior_textf, \
               prior_visualf, \
               prior_acouf,\
               textf, \
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               torch.LongTensor(prior_labels),\
               torch.LongTensor(emotion_shift),\
               vid    
    def __len__(self):
        return self.len
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<8 else pad_sequence(dat[i], True) if i<12 else dat[i].tolist() for i in dat]
