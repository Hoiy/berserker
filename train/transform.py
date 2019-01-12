import collections
import tensorflow as tf

def create_tokenizer(vocab_path):
    import tokenization
    return tokenization.FullTokenizer(
        vocab_file=vocab_path,
        do_lower_case=False
    )

def _to_text_and_truths(text):
    tokens = list(filter(None, text.split(' ')))
    text = ''.join(tokens)
    truths = [
        1. if i == len(token) - 1 else 0.
        for token in tokens
        for i in range(len(token))
    ]
    assert len(text) == len(truths)
    return text, truths


# Observation on bert tokenizer:
# 1. '[UNK]' for oov
# 2. Some tokens may be prefixed with '##'
# 3. Tokens may have longer than length 1 even without '##', e.g. numbers
# 4. Multiple consecutive oov may map to multiple or one '[UNK]' token
def preprocess(text, tokenizer):
    """Convert raw training / testing data to bert tokens format and map their truths.

    >>> preprocess('Ｂ７３７—３００  新世纪  ——  一  １１１１  ＫＫ·Ｄ  。  １２月  ３１日  。  １１００  。  ６—１２  。  Ｄ', create_tokenizer())
    (['[UNK]', '[UNK]', '３０', '##０', '新', '世', '纪', '[UNK]', '[UNK]', '一', '[UNK]', '·', '[UNK]', '。', '１２', '月', '３', '##１', '日', '。', '１１', '##００', '。', '６', '[UNK]', '１２', '。', '[UNK]'], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    """

    text, truths = _to_text_and_truths(text)
    bert_tokens = tokenizer.tokenize(text)
    bert_truths = []

    j = 0
    i = 0
    while i < len(bert_tokens):
        t = bert_tokens[i][2:] if bert_tokens[i][:2] == '##' else bert_tokens[i]
        l = len(t) if t != '[UNK]' else 1
        if t == text[j:j+l] or t == '[UNK]':
            bert_truths.append(truths[j+l-1])
            j = j + l
            i = i + 1
            continue

        # cannot match, previous token must be '[UNK]'
        assert i > 0 and bert_tokens[i-1] == '[UNK]', (i, text, bert_tokens)
        # assign truth value to the previous '[UNK]' token
        bert_truths[-1] = truths[j]
        j = j + 1


    assert len(bert_tokens) == len(bert_truths)
    return bert_tokens, bert_truths


def postprocess(text, bert_tokens, bert_truths, threshold, seperator='  '):
    """Convert raw training / testing data to bert tokens format and map their truths.

    >>> postprocess('Ｂ７３７—３００  新世纪  ——  一  １１１１  ＫＫ·Ｄ  。  １２月  ３１日  。  １１００  。  ６—１２  。  Ｄ', ['[UNK]', '[UNK]', '３０', '##０', '新', '世', '纪', '[UNK]', '[UNK]', '一', '[UNK]', '·', '[UNK]', '。', '１２', '月', '３', '##１', '日', '。', '１１', '##００', '。', '６', '[UNK]', '１２', '。', '[UNK]'], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 0.5)
    'Ｂ７３７—３００  新世纪  ——  一  １１１１ＫＫ·Ｄ  。  １２月  ３１日  。  １１００  。  ６—１２  。  Ｄ'
    """
    print(text, bert_tokens, bert_truths)
    assert len(bert_tokens) == len(bert_truths)
    text, truths = _to_text_and_truths(text)

    truths = []
    j = 0
    i = 0
    while i < len(bert_tokens):
        t = bert_tokens[i][2:] if bert_tokens[i][:2] == '##' else bert_tokens[i]
        l = len(t) if t != '[UNK]' else 1
        if t == text[j:j+l] or t == '[UNK]':
            for k in range(l-1):
                truths.append(0.)
            truths.append(bert_truths[i])
            j = j + l
            i = i + 1
            continue

        # cannot match, previous token must be '[UNK]'
        assert i > 1 and bert_tokens[i-1] == '[UNK]', (i, text, bert_tokens)
        # Assign truth value of '[UNK]' to only the last matching char
        truths[-1] = 0.
        truths.append(bert_truths[i-1])
        j = j + 1

    assert len(truths) == len(text), (len(truths), len(text))

    tokens = []
    for is_token_end, char in zip(truths, text):
        tokens.append(char)
        if is_token_end >= threshold:
            tokens.append(seperator)

    return ''.join(tokens).rstrip(seperator)


def _create_byte_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def _create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def text_to_tfexample(text, max_seq_length, tokenizer):
  tokens_a, tokens_a_truth = preprocess(text, tokenizer)
  assert len(tokens_a) == len(tokens_a_truth)
  bert_tokens_len = len(tokens_a)
  tokens_b = None

  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

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
  segment_ids = []
  truths = []

  tokens.append("[CLS]")
  segment_ids.append(0)
  truths.append(1.)

  for token, truth in zip(tokens_a, tokens_a_truth):
    tokens.append(token)
    segment_ids.append(0)
    truths.append(truth)

  tokens.append("[SEP]")
  segment_ids.append(0)
  truths.append(1.)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    truths.append(0.)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(truths) == max_seq_length

  features = collections.OrderedDict()
  features['text'] = _create_byte_feature(text.encode('utf-8'))
  features['bert_tokens_len'] = _create_int_feature([bert_tokens_len])
  features["input_ids"] = _create_int_feature(input_ids)
  features["input_mask"] = _create_int_feature(input_mask)
  features["segment_ids"] = _create_int_feature(segment_ids)
  features["truths"] = _create_float_feature(truths)

  return tf.train.Example(features=tf.train.Features(feature=features))


def feature_spec(seq_length):
    return {
        "text": tf.VarLenFeature(tf.string),
        "bert_tokens_len": tf.FixedLenFeature([1], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "truths": tf.FixedLenFeature([seq_length], tf.float32),
    }


if __name__ == "__main__":
    import doctest
    doctest.testmod()
