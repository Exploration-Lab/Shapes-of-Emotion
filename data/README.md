## Data
We couldn't provide the data/features pickle file because of space constraints. Please refer to points below to get the required feature files. 

### MOSEI

#### Categorial data CMU-MOSEI - Contains sentiment labels for all video ids and video(FACET)), audio(Opensmile) and text(Glove) features
Pickle file can be downloaded through this link https://github.com/amanshenoy/multilogue-net/raw/5d6b6ff8b1a26cf0762d6c1ca3a99917e881bf26/data/categorical.pkl

#### Emotion/sentiment labels creation
We extracted the emotion labels for videos from CMU-MultimodalSDK (https://github.com/A2Zadeh/CMU-MultimodalSDK). And created two files emotion_labels.pkl and sentiment_labels.pkl

#### Openface features for video
We extracted the openface features for CMU-MOSEI dataset form CMU-MultimodalSDK (https://github.com/A2Zadeh/CMU-MultimodalSDK)

#### Generate BERT and SBERT features
You can generate BERT and SBERT features for text using the bert-base-uncased model and trained siamese model respectively. We provide a script (create_vectors.py) does this task for you. 

We provide the training data for the siamese component of the model. 

### IEMOCAP
We provide the IEMOCAP video, audio and text features for both 4-label and 6-label. We also provide the training data for training the siamese component of the model

#### Generate BERT and SBERT features:
You can generate BERT and SBERT features for text using the bert-base-uncased model and trained siamese model respectively. We provide a script (create_vectors.py) which does this task for you. 

```
Note: Once the file are downloaded/extracted please refer to `config.py` to put these files to respective required locations.
```