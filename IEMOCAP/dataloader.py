import torch, pickle, pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class IEMOCAPCategorical(Dataset):
    def __init__(self, path, bert_vectors, siamese_vectors, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid) if x in self.videoIDs or x in self.testVid]


        self.bert_vectors = pickle.load(open(bert_vectors, 'rb'), encoding='latin1')
        self.siamese_vectors = pickle.load(open(siamese_vectors, 'rb'), encoding='latin1')
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
        visualf = torch.FloatTensor(self.videoVisual[vid])

        # print(textf.shape, acouf.shape, visualf.shape)
        prior_textf = torch.zeros(textf.size())
        prior_acouf = torch.zeros(acouf.size())
        prior_visualf = torch.zeros(visualf.size())

        prior_labels = []
        emotion_shift = []

        prior_utterance_feature = []
        qmask = self.videoSpeakers[vid]

        predecessor_label = 0
        # print(self.videoLabels[vid])
        for j in range(num_utterance):
            label_val = self.videoLabels[vid][j]
            if j!=0:
                prior_textf[j] = textf[j-1].clone()
                prior_acouf[j] = acouf[j-1].clone()
                prior_visualf[j] = visualf[j-1].clone()
                predecessor_label = self.videoLabels[vid][j-1]
                
            emotion_shift.append(1-((predecessor_label in [0,2,4] and label_val in [1,3,5]) or (label_val in [0,2,4] and predecessor_label in [1,3,5])))
            prior_labels.append(predecessor_label)
        
        # print(textf.shape, acouf.shape, visualf.shape)
        # print(prior_textf.shape, prior_acouf.shape, prior_visualf.shape)
        return bert_textf,\
               visualf,\
               torch.FloatTensor(self.videoAudio[vid]),\
               prior_textf, \
               prior_visualf, \
               prior_acouf,\
               textf, \
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
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
