from torch.utils.data import DataLoader
import torch
import math
from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers.readers.MultilogueNetReader import MultilogueNetReader
import sys
import os
import gzip
import csv
import numpy as np
from config import *
import argparse
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Trains a categorical model for sentiment data with 1 as positive sentiment and 0 as negative sentiment")
    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=3, metavar='E', help='number of epochs')
    parser.add_argument('--gpu', type=int, required=True, help='gpu core')
    parser.add_argument('--label_count', type=int, default = 4, help='Number of labels, either 4 or 6')
    parser.add_argument('--dataset', type=str, required = True, help='Either of three datastes [MOSEI, MELD, IEMOCAP]')
    parser.add_argument('--path_classifier_weights', type=str, required = True, help='Path to save weights of classifier network')
    parser.add_argument('--model_save_path', type=str, required = True, help='Path to save model')


    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    if args.dataset == "IEMOCAP" and args.label_count == 4:
        PATH = IEMOCAP_4
    elif args.dataset == "IEMOCAP" and args.label_count == 6:
        PATH = IEMOCAP_6
    elif args.datset == "MELD":
        PATH = MELD
    elif args.dataset == "MOSEI":
        PATH = MOSEI
    else:
        exit(0)

    train_dataset_path = PATH.SIAMESE_TRAIN
    dev_dataset_path = PATH.SIAMESE_TEST
    #You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    model_name = 'bert-base-uncased'

    # Read the dataset
    train_batch_size = args.batch_size

    # model_save_path = PATH.SIAMESE_MODEL
    model_save_path = args.model_save_path
    print(model_save_path)

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    train_reader = MultilogueNetReader(train_dataset_path)
    dev_reader = MultilogueNetReader(dev_dataset_path)

    train_samples = train_reader.get_examples()
    dev_samples = dev_reader.get_examples()

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=train_batch_size)

    train_loss = losses.SoftmaxLoss2(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2, weight = torch.FloatTensor(PATH.SIAMESE_WEIGHTS).cuda())
    train_loss.to('cuda')
    dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model = train_loss,  name='mlnet-dev')
    # Configure the training
    num_epochs = args.epochs

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    # Train the model
    dev_evaluator(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=200,
            warmup_steps=warmup_steps,
            output_path=model_save_path
            )

    # torch.save(train_loss.state_dict(), PATH.SIAMESE_CLASSIFIER)
    torch.save(train_loss.state_dict(), f"{args.path_classifier_weights}/siamese_classifier.chk")

    train_loss.load_state_dict(torch.load(f"{args.path_classifier_weights}/siamese_classifier.chk"))
    model = SentenceTransformer(model_save_path, device = 'cuda')
    test_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model = train_loss,  name='mlnet-dev')
    print("Accuracy on dev: ", test_evaluator(model, output_path=model_save_path))
