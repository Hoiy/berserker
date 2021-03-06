import os
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from trainer.transform import parse_truths, bert_input_to_tfexample
from berserker.transform import preprocess

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("assets_dir", "assets", "The assets directory generated by assets.py.")
flags.DEFINE_string("output_dir", "dataset", "The output directory.")
flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    config = {}
    for dataset in ['pku', 'msr', 'cityu', 'as']:
        print(os.path.join(FLAGS.assets_dir, 'icwb2-data', 'training', '%s_training.utf8'%dataset))
        config['train_%s'%dataset] = pd.read_csv(
            os.path.join(FLAGS.assets_dir, 'icwb2-data', 'training', '%s_training.utf8'%dataset),
            header=None, sep='^'
        )[0]
        print(os.path.join(FLAGS.assets_dir, 'icwb2-data', 'testing', '%s_test.utf8'%dataset))
        config['test_%s'%dataset] = pd.read_csv(
            os.path.join(FLAGS.assets_dir, 'icwb2-data', 'testing', '%s_test.utf8'%dataset),
            header=None, sep='^'
        )[0]

    for type, ser in config.items():
        outfile = os.path.join(FLAGS.output_dir, '%s_%s.tfrecords'%(type, FLAGS.max_seq_length))
        with tf.python_io.TFRecordWriter(outfile) as writer:
            tf.logging.info('Writing to %s...'%outfile)
            for text in tqdm(ser):
                text, truths = parse_truths(text)
                bert_inputs, _, _ = preprocess(text, FLAGS.max_seq_length, truths)
                for i in range(len(bert_inputs['input_ids'])):
                    writer.write(bert_input_to_tfexample(
                        input_ids = bert_inputs['input_ids'][i],
                        input_mask = bert_inputs['input_mask'][i],
                        segment_ids = bert_inputs['segment_ids'][i],
                        truths = bert_inputs['truths'][i]
                    ).SerializeToString())


if __name__ == "__main__":
  tf.app.run()
