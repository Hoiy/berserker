import os, collections
import tensorflow as tf
import berserker
from berserker.ext import tokenization

class BERT_Tokenizer(tokenization.FullTokenizer):
    def is_cleaned(self, ch):
        return ch.isspace() or len(self.basic_tokenizer._clean_text(ch)) == 0


_BERT_TOKENIZER = BERT_Tokenizer(
    vocab_file=os.path.join(berserker.ASSETS_PATH, 'vocab.txt'),
    do_lower_case=False
)


def convert_ids_to_token(ids):
    return _BERT_TOKENIZER.convert_ids_to_token(ids)


def compute_mapping(char_list, bert_tokens):
    """
    Create a many-to-one mapping from char_list to bert_tokens
    >>> compute_mapping(['Ｂ', '７', '３', '７', '—', '３', '０', '０', ' ', ' ', '新', '世', '纪'], ['[UNK]', '[UNK]', '３０', '##０', '新', '世', '纪'])
    {0: 0,
     1: 0,
     2: 0,
     3: 0,
     4: 1,
     5: 2,
     6: 2,
     7: 3,
     8: 3,
     9: 3,
     10: 4,
     11: 5,
     12: 6}
    """

    assert len(char_list) >= len(bert_tokens)

    i = 0 # Loop index for bert_tokens
    j = 0 # Loop index for char_list
    matched_len = [0] # A list storing the number of char matched for each bert tokens, the first element is a dummy
    while i < len(bert_tokens):
        # Loop invariant:
        assert j == sum(matched_len)
        assert len(matched_len) == 1 or matched_len[-1] > 0
        assert j >= 0 and j < len(char_list), (j, len(char_list), i, len(bert_tokens))

        # Invisible token will be mapped to the last available bert_tokens
        if _BERT_TOKENIZER.is_cleaned(char_list[j]):
            matched_len[-1] += 1
            j = j + 1
            continue

        # If current bert_token is '[UNK]'
        if bert_tokens[i] == '[UNK]':
            matched_len.append(1)
            j = j + matched_len[-1]
            i = i + 1
            continue

        # bert_tokens[i] is not '[UNK]' AND char_list[j] is not cleaned char,
        # len(bert_tokens[i]) maybe greater than 1
        bert_token = bert_tokens[i][2:] if bert_tokens[i][:2] == '##' else bert_tokens[i]
        text_segment = char_list[j]

        # Fetch the next len(bert_token) characters from text, ignoring bert cleaned char
        l = 1
        while len(text_segment) < len(bert_token):
            assert j+l < len(char_list), (char_list, bert_token, j, l)
            if not _BERT_TOKENIZER.is_cleaned(char_list[j+l]):
                text_segment += char_list[j+l]
            l += 1


        if bert_token == text_segment:
            matched_len.append(l)
            j = j + matched_len[-1]
            i = i + 1
            continue

        # This is an mismatch, perform a roll back until the last '[UNK]' bert token
        while True:
            i -= 1
            if bert_tokens[i] == '[UNK]':
                last_len = matched_len.pop()
                matched_len[-1] += 1
                j = sum(matched_len)
                break
            matched_len.pop()

    # Match the remaining char to the last bert token
    if sum(matched_len) < len(char_list):
        matched_len[-1] = len(char_list) - sum(matched_len[:-1])

    assert len(char_list) == sum(matched_len)

    # Convert matched_len to final mapping
    j = 0
    i = 0
    mapping = {}
    for j in range(len(matched_len)):
        for k in range(matched_len[j]):
            mapping[i] = j if j == 0 else j-1
            i += 1

    return mapping


def _to_unpadded_bert_inputs(text, truth):
    assert len(text) == len(truth)
    bert_tokens = _tokenizer.tokenize(text)

    # Reconstruct a mapping on how each character in the text map to the
    # output of bert tokenizer
    #
    # With this mapping, we construct a bert truth from raw text truth
    # This mapping is also used for postprocessing, where we map bert output
    # to prediction values for all character in the text


def _backward_map(mapping, outputs):
    max_index = {}
    for i, o in mapping.items():
        max_index[o] = max(max_index[o], i) if o in max_index else i

    inputs = [0.] * len(set(mapping.keys()))
    for o, i in max_index.items():
        inputs[i] = outputs[o]

    return inputs


def _forward_map(mapping, inputs):
    outputs = [0.] * len(set(mapping.values()))
    for i in range(len(inputs)):
        outputs[mapping[i]] = inputs[i]

    return outputs


def _unpad_bert_outputs(bert_input, bert_output):
    length = sum(bert_input["input_mask"]) - 2
    bert_tokens = _BERT_TOKENIZER.convert_ids_to_tokens(bert_input["input_ids"][1:1+length])
    bert_preds = bert_output["predictions"][1:1+length]
    return bert_tokens, bert_preds



def _pad_bert_inputs(tokens_a, tokens_a_truth, max_seq_length):
    assert len(tokens_a) == len(tokens_a_truth)
    assert len(tokens_a) <= max_seq_length - 2 # Account for [CLS] and [SEP] with "- 2"

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    truths = []

    tokens.append("[CLS]")
    truths.append(0.)

    for token, truth in zip(tokens_a, tokens_a_truth):
        tokens.append(token)
        truths.append(truth)

    tokens.append("[SEP]")
    truths.append(0.)

    input_ids = _BERT_TOKENIZER.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        truths.append(0.)

    segment_ids = [0] * max_seq_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(truths) == max_seq_length

    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "truths": truths
    }

def batch_preprocess(texts, max_seq_length):
    fields = ["input_ids", "input_mask", "segment_ids", "truths"]
    bert_inputs = {f: [] for f in fields}
    mappings = []
    sizes = []
    for text in texts:
        bert_input, mapping, size = preprocess(text, max_seq_length)
        for f in fields:
            bert_inputs[f] += bert_input[f]
        mappings.append(mapping)
        sizes.append(size)
    return bert_inputs, mappings, sizes


"""
Input: single text, Output: multiple bert_inputs
1. Unify input from training data and test data (with or without spaces)
2. Convert to BERT tokens
3. Compute a mapping from input to bert input
4. Chunk by max_seq_length into multiple bert inputs

"""
def preprocess(text, max_seq_length, truths=None):
    if truths is None:
        truths = [0.] * len(text)

    bert_tokens = _BERT_TOKENIZER.tokenize(text)
    mapping = compute_mapping([ch for ch in text], bert_tokens)
    bert_truths = _forward_map(mapping, truths)

    assert len(bert_tokens) == len(bert_truths)

    # chunking to batch input
    SEQ_LENGTH = max_seq_length - 2

    bert_inputs = []
    while len(bert_tokens) > 0:
        bert_inputs.append(_pad_bert_inputs(
            bert_tokens[:SEQ_LENGTH],
            bert_truths[:SEQ_LENGTH],
            max_seq_length
        ))

        bert_tokens = bert_tokens[SEQ_LENGTH:]
        bert_truths = bert_truths[SEQ_LENGTH:]

    return {
        "input_ids": [bi["input_ids"] for bi in bert_inputs],
        "input_mask": [bi["input_mask"] for bi in bert_inputs],
        "segment_ids": [bi["segment_ids"] for bi in bert_inputs],
        "truths": [bi["truths"] for bi in bert_inputs]
    }, mapping, len(bert_inputs)


def batch_postprocessing(texts, mappings, sizes, bert_inputs, bert_outputs, max_seq_length, threshold=0.5):
    fields = ["input_ids", "input_mask", "segment_ids", "truths"]
    results = []
    i = 0
    for text, mapping, size in zip(texts, mappings, sizes):
        bi = []
        for j in range(size):
            bi.append({
                f: bert_inputs[f][i+j] for f in fields
            })
        bo = bert_outputs[i:i+size]
        print(bo)
        results.append(postprocess(text, mapping, bi, bo, threshold))
        i += size
    return results


def postprocess(text, mapping, bert_inputs, bert_outputs, threshold=0.5):
    assert len(bert_inputs) == len(bert_outputs)
    bert_preds = []
    for bert_input, bert_output in zip(bert_inputs, bert_outputs):
        bert_token, bert_pred = _unpad_bert_outputs(bert_input, bert_output)
        bert_preds += bert_pred.tolist()

    assert len(bert_preds) == len(set(mapping.values())), (len(bert_preds), len(set(mapping.values())))
    preds = _backward_map(mapping, bert_preds)
    assert len(text) == len(preds)
    result = ""
    for ch, pred in zip(text, preds):
        result += ch
        if pred >= threshold:
            result += " "
    return result.split(" ")





    # return {
    #     'input_ids': [bi[0] for bi in bert_inputs],
    #     'input_mask': [bi[1] for bi in bert_inputs],
    #     'segment_ids': [bi[2] for bi in bert_inputs],
    #     'truths': [bi[3] for bi in bert_inputs]
    # }


# # Observation on bert tokenizer:
# # 1. '[UNK]' for oov
# # 2. Some tokens may be prefixed with '##'
# # 3. Tokens may have longer than length 1 even without '##', e.g. numbers
# # 4. Multiple consecutive oov may map to multiple or one '[UNK]' token
# def preprocess(text, tokenizer):
#     """Convert raw training / testing data to bert tokens format and map their truths.
#
#     >>> preprocess('Ｂ７３７—３００  新世纪  ——  一  １１１１  ＫＫ·Ｄ  。  １２月  ３１日  。  １１００  。  ６—１２  。  Ｄ', create_tokenizer())
#     (['[UNK]', '[UNK]', '３０', '##０', '新', '世', '纪', '[UNK]', '[UNK]', '一', '[UNK]', '·', '[UNK]', '。', '１２', '月', '３', '##１', '日', '。', '１１', '##００', '。', '６', '[UNK]', '１２', '。', '[UNK]'], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
#     """
#
#     text, truths = _to_text_and_truths(text)
#     bert_tokens = tokenizer.tokenize(text)
#     bert_truths = []
#
#     j = 0
#     i = 0
#     while i < len(bert_tokens):
#         t = bert_tokens[i][2:] if bert_tokens[i][:2] == '##' else bert_tokens[i]
#         l = len(t) if t != '[UNK]' else 1
#         if t == text[j:j+l] or t == '[UNK]':
#             bert_truths.append(truths[j+l-1])
#             j = j + l
#             i = i + 1
#             continue
#
#         # cannot match, previous token must be '[UNK]'
#         assert i > 0 and bert_tokens[i-1] == '[UNK]', (i, text, bert_tokens)
#         # assign truth value to the previous '[UNK]' token
#         bert_truths[-1] = truths[j]
#         j = j + 1
#
#
#     assert len(bert_tokens) == len(bert_truths)
#     return bert_tokens, bert_truths
#
#
# def postprocess(text, bert_tokens, bert_truths, threshold, seperator='  '):
#     """Convert raw training / testing data to bert tokens format and map their truths.
#
#     >>> postprocess('Ｂ７３７—３００  新世纪  ——  一  １１１１  ＫＫ·Ｄ  。  １２月  ３１日  。  １１００  。  ６—１２  。  Ｄ', ['[UNK]', '[UNK]', '３０', '##０', '新', '世', '纪', '[UNK]', '[UNK]', '一', '[UNK]', '·', '[UNK]', '。', '１２', '月', '３', '##１', '日', '。', '１１', '##００', '。', '６', '[UNK]', '１２', '。', '[UNK]'], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 0.5)
#     'Ｂ７３７—３００  新世纪  ——  一  １１１１ＫＫ·Ｄ  。  １２月  ３１日  。  １１００  。  ６—１２  。  Ｄ'
#     """
#     assert len(bert_tokens) == len(bert_truths)
#     text, truths = _to_text_and_truths(text)
#
#     truths = []
#     j = 0
#     i = 0
#     while i < len(bert_tokens):
#         t = bert_tokens[i][2:] if bert_tokens[i][:2] == '##' else bert_tokens[i]
#         l = len(t) if t != '[UNK]' else 1
#         if t == text[j:j+l] or t == '[UNK]':
#             for k in range(l-1):
#                 truths.append(0.)
#             truths.append(bert_truths[i])
#             j = j + l
#             i = i + 1
#             continue
#
#         # cannot match, previous token must be '[UNK]'
#         assert i > 1 and bert_tokens[i-1] == '[UNK]', (i, j, text[j-2:j+3], bert_tokens[i-2:i+3])
#         # Assign truth value of '[UNK]' to only the last matching char
#         truths[-1] = 0.
#         truths.append(bert_truths[i-1])
#         j = j + 1
#
#     assert len(truths) == len(text), (len(truths), len(text), text, bert_truths, bert_tokens)
#
#     tokens = []
#     for is_token_end, char in zip(truths, text):
#         tokens.append(char)
#         if is_token_end >= threshold:
#             tokens.append(seperator)
#
#     return list(filter(None, ''.join(tokens).rstrip(seperator).split('  ')))
#
#
# def _create_byte_feature(values):
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
#
# def _create_int_feature(values):
#   feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
#   return feature
#
#
# def _create_float_feature(values):
#   feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
#   return feature
#
#

#
#
# def bert_inputs_to_tfexample(input_ids, input_mask, segment_ids, truths):
#   features = collections.OrderedDict()
#   # features['text'] = _create_byte_feature(text.encode('utf-8'))
#   # features['bert_tokens_len'] = _create_int_feature([bert_tokens_len])
#   features["input_ids"] = _create_int_feature(input_ids)
#   features["input_mask"] = _create_int_feature(input_mask)
#   features["segment_ids"] = _create_int_feature(segment_ids)
#   features["truths"] = _create_float_feature(truths)
#
#   return tf.train.Example(features=tf.train.Features(feature=features))
#
#
# def text_to_tfexample(text, max_seq_length, tokenizer):
#     return bert_inputs_to_tfexample(*text_to_bert_inputs(text, max_seq_length, tokenizer))
#
#
def feature_spec(seq_length):
    return {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "truths": tf.FixedLenFeature([seq_length], tf.float32),
    }
#
#
# # def model_prediction_to_result(predictions):
# #     text = predictions['text'].eval().values[0].decode('utf-8')
# #     bert_tokens_len = features['bert_tokens_len'].eval()[0]
# #     bert_tokens = tokenizer.convert_ids_to_tokens(features['input_ids'].eval()[1:bert_tokens_len+1])
# #     bert_truths = features['truths'].eval()[1:bert_tokens_len+1]
# #
# #     print("Final Result:", postprocess(text, bert_tokens, bert_truths, 0.5))
#
#
#
# if __name__ == "__main__":
#     pass
