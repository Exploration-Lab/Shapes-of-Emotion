# emo-prediction-with-emo-shift
A step by step series of examples that tell you how to get the code running

Unzip the file. 

```
unzip code.zip
```
#### Create and start a virtual environment
#### Using conda
```
conda create -n env python=3.6

source activate env
```
#### Install the project dependencies:
```
cd code
pip3 install -r requirements.txt
```

### Running the code

#### Command for training the siamese network
```
python train_siamese.py --gpu <core_no> --model_save_path <model_path> --dataset <dataset_name>  --label_count <labels_count> 
```
An example commad is given below
```
python train_siamese.py --gpu 0 --model_save_path siamese_model/IEMOCAP --dataset IEMOCAP --label_count 6 

```
#### Command for training the emotion classification model
##### MOSEI
###### Sentiment Classification
```
python MOSEI/train_sentiment.py --gpu <core_no> --model_save_name <model_path> 
```
###### Emotion Classification 
```
python MOSEI/train_emotion.py --gpu <core_no> --model_save_name <model_path> --emotion <emotion>
```

##### IEMOCAP
###### Emotion Classification
We provide data and code for two vaiants of IEMOCAP - 4 label and 6 label. 
```
python IEMOCAP/train.py --gpu <core_no> --model_save_name <model_path> --label_count <labels> 
```

##### MELD
###### Emotion Classification 
```
python MELD/train.py --gpu <core_no> --model_save_name <model_path>
```