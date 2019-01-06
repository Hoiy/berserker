import os

TOKENIZER_VOCAB = os.path.join('assets', 'chinese_L-12_H-768_A-12', 'vocab.txt')
DATASET_PATH = os.path.join('assets', 'icwb2-data')

TRAINING_DATA_PATH = os.path.join(DATASET_PATH, 'training', 'pku_training.utf8')
TESTING_DATA_PATH = os.path.join(DATASET_PATH, 'testing', 'pku_test.utf8')
