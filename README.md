## Shapes of Emotions: Multimodal Emotion Recognition in Conversations viaEmotion Shifts
A step by step series of examples that tell you how to get the code running

Unzip the file
```
unzip code.zip
```
#### Create and start a virtual environment (using conda)
```
conda create -n env python=3.6

source activate env
```
#### Install the project dependencies:
```
cd code
pip3 install -r requirements.txt
```
#### Get Data files required for the project
Few data files for the project are provided with the zip itself but due to space constraints we could not upload all the data files. You can find the instructions to get all the data file in `data/README.md`, once done with this step you can proceed to next step of running the code. 

### Running the code

#### Setup python-path for using sentence-transformer
```
export PYTHONPATH="$PWD/sentence-transformers:$PYTHONPATH"
```

#### Command for training the siamese network
```
python train_siamese.py --gpu <core_no> --model_save_path <model_path> --dataset <dataset_name>  --label_count <labels_count> --path_classifier_weights <path_to_save_classifier_weights>
```
An example commad is given below
```
python train_siamese.py --gpu 0 --model_save_path data/IEMOCAP/six_class/sbert_model --dataset IEMOCAP --label_count 6 --path_classifier_weights data/IEMOCAP/six_class/

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
