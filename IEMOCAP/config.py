from dataclasses import dataclass

@dataclass
class paths:
    BERT_VECTORS: str
    SBERT_VECTORS: str
    SIAMESE_MODEL: str
    SIAMESE_CLASSIFIER: str
    CATEGORICAL_DATA: str


paths_4 = paths(BERT_VECTORS = "../data/IEMOCAP/four_class/bert_vectors.p", SBERT_VECTORS ="../data/IEMOCAP/four_class/sbert_vectors.p", SIAMESE_MODEL="../data/IEMOCAP/four_class/sbert_model", SIAMESE_CLASSIFIER = "../data/IEMOCAP/four_class/siamese_classifier.chk", CATEGORICAL_DATA= "../data/IEMOCAP/four_class/IEMOCAP_features.pkl")
paths_6 = paths(BERT_VECTORS = "../data/IEMOCAP/six_class/bert_vectors.p", SBERT_VECTORS ="../data/IEMOCAP/six_class/sbert_vectors.p", SIAMESE_MODEL="../data/IEMOCAP/six_class/sbert_model", SIAMESE_CLASSIFIER = "../data/IEMOCAP/six_class/siamese_classifier.chk", CATEGORICAL_DATA= "../data/IEMOCAP/six_class/IEMOCAP_features.pkl")

