"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import torch
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator,BinaryClassificationEvaluator, TripletEvaluator
from sentence_transformers.readers.MultilogueNetTripletReader import MultilogueNetTripletReader
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import numpy as np

np.random.seed(393)
torch.manual_seed(393)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.set_device(2)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
train_dataset_path = '/data/harsh/UGP/multilogue-net/siamese_triplet_train_new.pickle'
dev_dataset_path = '/data/harsh/UGP/multilogue-net/siamese_triplet_test_new.pickle'

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

# Read the dataset
train_batch_size = 4

#+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"
model_save_path = 'output/training_triplet_new_data_semi_hard_'+model_name.replace("/", "-")
print(model_save_path)

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



train_reader = MultilogueNetTripletReader(train_dataset_path)
dev_reader = MultilogueNetTripletReader(dev_dataset_path)

train_samples = train_reader.get_examples()
dev_samples = dev_reader.get_examples()

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# train_loss = losses.TripletLoss(model=model)
# train_loss = losses.BatchAllTripletLoss(model=model)
# train_loss = losses.BatchHardTripletLoss(model=model)
# train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
train_loss = losses.BatchSemiHardTripletLoss(model=model)

dev_evaluator = TripletEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# Configure the training
num_epochs = 10

logging.info('Performance before training')
model.evaluate(dev_evaluator)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
# ##############################################################################

# test_samples = []
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'test':
#             score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
#             test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

model = SentenceTransformer(model_save_path)
test_evaluator = TripletEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)
# dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model = train_loss,  name='sts-dev')
