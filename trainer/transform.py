import collections
import tensorflow as tf

def parse_truths(text):
    """
    Convert text into text and truths, a value of 1.0 indicates that it is the end of a token.
    >>> _parse_truths("迈向  充满  希望  的  新  世纪  ")
    ('迈向充满希望的新世纪', [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    """
    tokens = list(filter(None, text.replace('\u3000', ' ').split(' ')))
    text = ''.join(tokens)
    truths = [
        1. if i == len(token) - 1 else 0.
        for token in tokens
        for i in range(len(token))
    ]
    assert len(text) == len(truths)
    return text, truths


def _create_byte_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def _create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def bert_input_to_tfexample(input_ids, input_mask, segment_ids, truths):
  features = collections.OrderedDict()
  features["input_ids"] = _create_int_feature(input_ids)
  features["input_mask"] = _create_int_feature(input_mask)
  features["segment_ids"] = _create_int_feature(segment_ids)
  features["truths"] = _create_float_feature(truths)

  return tf.train.Example(features=tf.train.Features(feature=features))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
