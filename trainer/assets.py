import os
import tensorflow as tf
from berserker.utils import maybe_download_unzip, maybe_git_clone

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "assets", "The output directory.")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)


    maybe_download_unzip(
        'http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip',
        FLAGS.output_dir,
        verbose=True
    )

    maybe_download_unzip(
        'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip',
        FLAGS.output_dir,
        verbose=True
    )


if __name__ == "__main__":
    tf.app.run()
