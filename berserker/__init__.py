from .utils import maybe_download_unzip
from .tokenization import FullTokenizer
from pathlib import Path
from .transform import text_to_bert_inputs, preprocess, postprocess
import tensorflow as tf
import numpy as np


_assets_path = Path(__file__).parent / 'assets'
_models_path = Path(__file__).parent / 'models'


MAX_SEQ_LENGTH = 512
SEQ_LENGTH = MAX_SEQ_LENGTH - 2


def load_model(model_name=None, verbose=True, force_download=False):
    maybe_download_unzip(
        'https://github.com/Hoiy/berserker/releases/download/v0.1-alpha/1547563491.zip',
        _models_path,
        verbose,
        force_download
    )


def tokenize(text):
    load_model()
    tokenizer = FullTokenizer(
        vocab_file=_assets_path / 'vocab.txt',
        do_lower_case=False
    )

    bert_inputs_lens = []
    bert_inputs = []
    temp = text
    while len(temp) > 0:
        bert_input = text_to_bert_inputs(temp[:SEQ_LENGTH], MAX_SEQ_LENGTH, tokenizer)
        bert_inputs_lens.append(len(preprocess(temp[:SEQ_LENGTH], tokenizer)[0]))
        bert_inputs.append(bert_input)
        temp = temp[SEQ_LENGTH:]

    berserker = tf.contrib.predictor.from_saved_model(
        str(_models_path / '1547563491')
    )
    output = berserker({
        'input_ids': [bi[0] for bi in bert_inputs],
        'input_mask': [bi[1] for bi in bert_inputs],
        'segment_ids': [bi[2] for bi in bert_inputs],
        'truths': [bi[3] for bi in bert_inputs]
    })

    results = output['predictions']

    results_itr = iter(results)
    bert_inputs_itr = iter(bert_inputs)
    bert_inputs_lens_itr = iter(bert_inputs_lens)

    prediction = np.array([])
    bert_tokens = []
    temp = text

    while len(temp) > 0:
        (input_ids, _, _, _) = next(bert_inputs_itr)
        bert_inputs_len = next(bert_inputs_lens_itr)
        result = next(results_itr)
        prediction = np.concatenate((prediction, result[1:1+bert_inputs_len]))
        bert_tokens += tokenizer.convert_ids_to_tokens(input_ids[1:1+bert_inputs_len])
        temp = temp[SEQ_LENGTH:]

    return postprocess(text, bert_tokens, prediction, threshold=0.5)
