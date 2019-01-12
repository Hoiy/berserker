import tensorflow as tf
from input import feature_spec

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("tfrecords_file", "dataset/train_128.tfrecords", "The tfrecords file to be inspected.")
flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
flags.DEFINE_integer("display_rows", 3, "Number of rows to be displayed.")

def view_tfrecord(file_name, feature_spec, display_rows=5):
    """
    >>> view_tfrecord('./dataset/val_128.tfrecords', 128)
    """
    with tf.Session() as sess:
        i = 0
        for example in tf.python_io.tf_record_iterator(file_name):
            if i >= display_rows:
                break
            i+=1
            features = tf.parse_single_example(example, features=feature_spec)
            print('Example %d:'%i)
            for feature_name, tensor in features.items():
                print(feature_name, features[feature_name].eval())


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    view_tfrecord(
        FLAGS.tfrecords_file,
        feature_spec(FLAGS.max_seq_length),
        FLAGS.display_rows
    )


if __name__ == "__main__":
  tf.app.run()
