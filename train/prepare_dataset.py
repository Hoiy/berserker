import os
import tensorflow as tf
import pandas as pd
from train.config import TRAINING_DATA_PATH, TESTING_DATA_PATH
from train.transform import create_tokenizer, text_to_tfexample
from tqdm import tqdm

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    tokenizer = create_tokenizer()
    config = {
        'train': TRAINING_DATA_PATH,
        'test': TESTING_DATA_PATH
    }
    for type, src in config.items():
        outfile = os.path.join(FLAGS.output_dir, '%s.tfrecords'%type)
        with tf.python_io.TFRecordWriter(outfile) as writer:
            tf.logging.info('Writing to %s...'%outfile)
            for text in tqdm(pd.read_csv(src, header=None)[0]):
                writer.write(text_to_tfexample(text, FLAGS.max_seq_length, tokenizer).SerializeToString())


if __name__ == "__main__":
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
