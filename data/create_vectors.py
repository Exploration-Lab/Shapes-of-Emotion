from sentence_transformers import SentenceTransformer
import torch
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Creates BERT/SBERT embeddings for text modality given a pre-trained language model and feature file containing sentences")
    parser.add_argument('--model_location', type=str, required=False, help='Location of model file')
    parser.add_argument('--embeddings', type=str, required = True, help='Either of two [BERT, SBERT]')
    parser.add_argument('--gpu', type=int, required=True, help='gpu core')
    parser.add_argument('--feature_file', type=str, required=True, help='Provide location of feature file containing the sentences of the dataset')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    if args.embeddings == "BERT":
        model = SentenceTransformer('bert-base-uncased')
    elif args.embeddings == "SBERT":
        model = SentenceTransformer(args.model_location)
    else:
        print("Given arguments not valid")
        exit(0)

    videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(open(args.feature_file, 'rb'), encoding='latin1')
    texts = []
    video_ids = []
    for vidID in videoIDs:
        for ind, text in enumerate(videoSentence[vidID]):
            video_ids.append(vidID + "[" + str(ind) + "]")
            texts.append(text)

    linguistic = model.encode(texts)
    dict_ = {}
    for i, vectors in enumerate(linguistic):
        dict_[video_ids[i]] = linguistic[i]

    embeddings_type = args.embeddings
    pickle.dump(dict_, open(embeddings.lower() + '_vectors.p','wb'))

