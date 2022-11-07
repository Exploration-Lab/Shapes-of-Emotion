## Shapes of Emotions: Multimodal Emotion Recognition in Conversations viaEmotion Shifts

The repository contains the full codebase of experiments and results of the paper "Shapes of Emotions: Multimodal Emotion Recognition in Conversations viaEmotion Shifts"

If you use the models proposed in the paper, please cite the paper (citation given below).

# Models and Code

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

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

The RR corpus and software follows [CC-BY-NC](CC-BY-NC) license. Thus, users can share and adapt our dataset if they give credit to us and do not use our dataset for any commercial purposes.


## Citation

```
@inproceedings{bansal-etal-2022-shapes,
    title = "Shapes of Emotions: Multimodal Emotion Recognition in Conversations via Emotion Shifts",
    author = "Bansal, Keshav  and
      Agarwal, Harsh  and
      Joshi, Abhinav  and
      Modi, Ashutosh",
    booktitle = "Proceedings of the First Workshop on Performance and Interpretability Evaluations of Multimodal, Multipurpose, Massive-Scale Models",
    month = oct,
    year = "2022",
    address = "Virtual",
    publisher = "International Conference on Computational Linguistics",
    url = "https://aclanthology.org/2022.mmmpie-1.6",
    pages = "44--56",
    abstract = "Emotion Recognition in Conversations (ERC) is an important and active research area. Recent work has shown the benefits of using multiple modalities (e.g., text, audio, and video) for the ERC task. In a conversation, participants tend to maintain a particular emotional state unless some stimuli evokes a change. There is a continuous ebb and flow of emotions in a conversation. Inspired by this observation, we propose a multimodal ERC model and augment it with an emotion-shift component that improves performance. The proposed emotion-shift component is modular and can be added to any existing multimodal ERC model (with a few modifications). We experiment with different variants of the model, and results show that the inclusion of emotion shift signal helps the model to outperform existing models for ERC on MOSEI and IEMOCAP datasets.",
}
```
