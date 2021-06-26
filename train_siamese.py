"""
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
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator,BinaryClassificationEvaluator,LabelAccuracyEvaluator
from sentence_transformers.readers.MultilogueNetReader import MultilogueNetReader
import sys
import os
import gzip
import csv
import numpy as np
from config import *

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Trains a categorical model for sentiment data with 1 as positive sentiment and 0 as negative sentiment")
    parser.add_argument('--batch_size', type=int, default=128, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=3, metavar='E', help='number of epochs')
    parser.add_argument('--gpu', type=int, required=True, help='gpu core')
    parser.add_argument('--label_count', type=int, required=True, help='Number of labels, either 4 or 6')
    parser.add_argument('--dataset', type=str, required = True, help='Either of three datastes [MOSEI, MELD, IEMOCAP]')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    if args.dataset == "IEMOCAP" and args.label_count == 4:
        paths = IEMOCAP_4
    elif args.dataset == "IEMOCAP" and args.label_count == 6:
        paths = IEMOCAP_6
    elif args.datset == "MELD":
        paths = MELD
    elif args.dataset == "MOSEI":
        paths = MOSEI
    else:
        exit(0)

    train_dataset_path = paths.SIAMESE_TRAIN
    dev_dataset_path = paths.SIAMESE_TEST
    #You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

    # Read the dataset
    train_batch_size = args.batch_size

    #+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"
    model_save_path = paths.SIAMESE_MODEL
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

    train_loss = losses.SoftmaxLoss2(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2, weight = torch.FloatTensor(paths.SIAMESE_WEIGHTS).cuda())
    train_loss.to('cuda')
    # dev_evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
    dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model = train_loss,  name='mlnet-dev')
    # Configure the training
    num_epochs = args.epochs

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    # logging.info("Warmup-steps: {}".format(warmup_steps))

    # torch.save(train_loss.state_dict(), 'loss_function.chk')
    # train_loss.load_state_dict(torch.load('loss_function.chk'))

    # Train the model
    dev_evaluator(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path
            )

    torch.save(train_loss.state_dict(), paths.SIAMESE_CLASSIFIER)

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
    # train_loss.load_state_dict(torch.load('../IEMOCAP/loss_function.chk'))
    # model = SentenceTransformer(model_save_path, device = 'cuda')
    # test_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model = train_loss,  name='mlnet-dev')


    # # test_evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-test')
    # test_evaluator(model, output_path=model_save_path)
    # # dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model = train_loss,  name='sts-dev')
