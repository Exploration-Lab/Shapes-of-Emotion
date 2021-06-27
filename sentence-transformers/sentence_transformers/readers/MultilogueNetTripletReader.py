from . import InputExample
import csv
import gzip
import os
import pickle

class MultilogueNetTripletReader(object):
    """
    Reads in the a Triplet Dataset: Each line contains (at least) 3 columns, one anchor column (s1),
    one positive example (s2) and one negative example (s3)
    """
    def __init__(self, filename):
        self.dataset_file = pickle.load(open(filename, 'rb'))

    def get_examples(self):
        """

        """
        examples = []
        for sentence_a, sentence_b, sentence_c in self.dataset_file:
            # print(sentence_a, sentence_b, label)
            examples.append(InputExample(texts=[sentence_a, sentence_b, sentence_c]))


        return examples