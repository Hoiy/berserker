from io import BytesIO
from zipfile import ZipFile
import tempfile
import requests
import subprocess
import os
import ntpath
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "assets", "The output directory.")

def maybe_download_unzip(url, dst_path):
    dir = os.path.join(dst_path, os.path.splitext(ntpath.basename(url))[0])
    if os.path.exists(dir):
        tf.logging.info('Directory %s already exists, skipping download...'%dir)
        return
    tf.logging.info('Downloading %s...'%url)
    req = requests.get(url)
    zipfile = ZipFile(BytesIO(req.content))
    zipfile.extractall(dst_path)
    return

def maybe_git_clone(url, dst_path):
    repo_name = os.path.splitext(ntpath.basename(url))[0]
    dir = os.path.join(dst_path, repo_name)
    if os.path.exists(dir):
        tf.logging.info('Directory %s already exists, skipping download...'%dir)
        return

    tf.logging.info('Cloning %s...'%url)
    subprocess.call(['git', 'clone',
        url,
        os.path.join(dst_path, repo_name)
    ])
    return


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)


    maybe_download_unzip(
        'http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip',
        FLAGS.output_dir
    )

    maybe_download_unzip(
        'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip',
        FLAGS.output_dir
    )

    maybe_git_clone(
        'https://github.com/google-research/bert',
        FLAGS.output_dir
    )


if __name__ == "__main__":
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
