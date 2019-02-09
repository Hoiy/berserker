from berserker.utils import maybe_download_unzip
from pathlib import Path
import tensorflow as tf
import numpy as np

ASSETS_PATH = str(Path(__file__).parent / 'assets')
_models_path = Path(__file__).parent / 'models'

from berserker.transform import batch_preprocess, batch_postprocess

MAX_SEQ_LENGTH = 512
SEQ_LENGTH = MAX_SEQ_LENGTH - 2
BATCH_SIZE = 8

def load_model(model_name=None, verbose=True, force_download=False):
    maybe_download_unzip(
        'https://github.com/Hoiy/berserker/releases/download/v0.1-alpha/1547563491.zip',
        _models_path,
        verbose,
        force_download
    )


def tokenize(text):
    load_model()
    texts = [text]
    bert_inputs, mappings, sizes = batch_preprocess(texts, MAX_SEQ_LENGTH, BATCH_SIZE)

    berserker = tf.contrib.predictor.from_saved_model(
        str(_models_path / '1547563491')
    )
    bert_outputs = berserker(bert_inputs)
    bert_outputs = [{'predictions': bo} for bo in bert_outputs['predictions']]

    return batch_postprocess(texts, mappings, sizes, bert_inputs, bert_outputs, MAX_SEQ_LENGTH)[0]
