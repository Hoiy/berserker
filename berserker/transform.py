import os, collections
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

def batch_preprocess(texts, max_seq_length, batch_size):
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

    # pad to batch_size
    while len(bert_inputs["input_ids"]) % batch_size != 0:
        for f in fields:
            bert_inputs[f].append([0] * max_seq_length)
    return bert_inputs, mappings, sizes

# Input: single text, Output: multiple bert_inputs
# 1. Unify input from training data and test data (with or without spaces)
# 2. Convert to BERT tokens
# 3. Compute a mapping from input to bert input
# 4. Chunk by max_seq_length into multiple bert inputs

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
    while len(bert_tokens) > 0 or len(bert_inputs) == 0:
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


def batch_postprocess(texts, mappings, sizes, bert_inputs, bert_outputs, max_seq_length, threshold=0.5):
    assert len(bert_inputs["input_ids"]) == len(bert_outputs), (len(bert_inputs["input_ids"]), len(bert_outputs))
    results = []
    i = 0
    for text, mapping, size in zip(texts, mappings, sizes):
        bi = [{k: bert_inputs[k][i+j] for k in bert_inputs.keys()} for j in range(size)]
        bo = bert_outputs[i:i+size]
        results.append(postprocess(text, mapping, bi, bo, threshold))
        i += size
    return results


def postprocess(text, mapping, bert_inputs, bert_outputs, threshold=0.5):
    assert len(bert_inputs) == len(bert_outputs), (len(bert_inputs), len(bert_outputs))
    bert_preds = []
    for bert_input, bert_output in zip(bert_inputs, bert_outputs):
        bert_token, bert_pred = _unpad_bert_outputs(bert_input, bert_output)
        bert_preds += bert_pred.tolist()

    assert len(bert_preds) == len(set(mapping.values())), (len(bert_preds), len(set(mapping.values())))
    preds = _backward_map(mapping, bert_preds)
    assert len(text) == len(preds), (text, preds)
    result = ""
    for ch, pred in zip(text, preds):
        result += ch
        if pred >= threshold:
            result += " "
    return list(filter(None, result.split(" ")))
