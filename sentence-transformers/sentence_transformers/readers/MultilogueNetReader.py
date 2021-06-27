from . import InputExample
import csv
import gzip
import os
import pickle


class MultilogueNetReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self, filename):
        self.dataset_file = pickle.load(open(filename, 'rb'))

    def get_examples(self):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """
        examples = []
        for sentence_a, sentence_b, label in self.dataset_file:
            # print(sentence_a, sentence_b, label)
            examples.append(InputExample(texts=[sentence_a, sentence_b], label=label))


        return examples

    # @staticmethod
    # def get_labels():
    #     return {"contradiction": 0, "entailment": 1, "neutral": 2}

    # def get_num_labels(self):
    #     return len(self.get_labels())

    # def map_label(self, label):
    #     return self.get_labels()[label.strip().lower()]