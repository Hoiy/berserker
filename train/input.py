import tensorflow as tf
from transform import feature_spec, preprocess, text_to_bert_inputs

def _deserialize(serialized, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(serialized, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def input_fn_builder(input_file, seq_length, shuffle, repeat, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if shuffle:
      d = d.shuffle(buffer_size=1024)
    if repeat:
      d = d.repeat()

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda serialized: _deserialize(serialized, feature_spec(seq_length)),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
    return d

  return input_fn


def serving_input_fn_builder(seq_length, batch_size):
    def serving_input_receiver_fn():
      """An input receiver that expects a serialized tf.Example."""
      serialized_tf_example = tf.placeholder(dtype=tf.string,
                                             shape=[batch_size],
                                             name='input_example_tensor')
      receiver_tensors = {'examples': serialized_tf_example}
      features = tf.parse_example(serialized_tf_example, feature_spec(seq_length))
      return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    return serving_input_receiver_fn


# TODO: padding insteads of drop_remainder
def predict_input_fn_builder(bert_inputs, seq_length, tokenizer, drop_remainder=False):
  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_truths = []

  for input_ids, input_mask, segment_ids, truths in bert_inputs:
      all_input_ids.append(input_ids)
      all_input_mask.append(input_mask)
      all_segment_ids.append(segment_ids)
      all_truths.append(truths)

  num_examples = len(bert_inputs)

  def input_fn(params):
      d = tf.data.Dataset.from_tensor_slices({
          "input_ids":
              tf.constant(
                  all_input_ids, shape=[num_examples, seq_length],
                  dtype=tf.int32),
          "input_mask":
              tf.constant(
                  all_input_mask,
                  shape=[num_examples, seq_length],
                  dtype=tf.int32),
          "segment_ids":
              tf.constant(
                  all_segment_ids,
                  shape=[num_examples, seq_length],
                  dtype=tf.int32),
          "truths":
              tf.constant(
                  all_truths,
                  shape=[num_examples, seq_length],
                  dtype=tf.float32),
      })
      d = d.batch(batch_size=params['batch_size'], drop_remainder=drop_remainder)
      return d

  return input_fn

def serving_input_fn_builder(seq_length):
   return tf.estimator.export.build_raw_serving_input_receiver_fn({
    'input_ids': tf.placeholder(
        shape=[None, seq_length],
        dtype=tf.int32,
        name='input_ids_ph'
    ),
    'input_mask': tf.placeholder(
        shape=[None, seq_length],
        dtype=tf.int32,
        name='input_mask_ph'
    ),
    'segment_ids': tf.placeholder(
        shape=[None, seq_length],
        dtype=tf.int32,
        name='segment_ids_ph'
    ),
    'truths': tf.placeholder(
        shape=[None, seq_length],
        dtype=tf.float32,
        name='truths_ph'
    )
  })
