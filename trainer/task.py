import os
import tensorflow as tf
from trainer.input import input_fn_builder, serving_input_fn_builder, predict_input_fn_builder
from trainer.model import model_fn_builder
from trainer.ext import modeling
from berserker.transform import batch_preprocess, batch_postprocessing
from tqdm import tqdm
import sys

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("assets_dir", "assets", "The assets directory generated by assets.py.")
flags.DEFINE_string("checkpoint_dir", '/tmp/ckpt', "The directory for storing model check points.")
flags.DEFINE_string("gs_bert_model_ch_dir", 'gs://berserker/repo/assets/chinese_L-12_H-768_A-12', "A google storage path to unzipped BERT chinese_L-12_H-768_A-12 model.")
flags.DEFINE_string("train_file", "dataset/train_pku_512.tfrecords", "The training file output by dataset.py.")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
flags.DEFINE_integer("batch_size", 64, "The training, validation and prediction batch size.")

flags.DEFINE_bool("do_train", False, "Train the model.")
flags.DEFINE_float("learning_rate", 2e-5, "The learning rate.")
flags.DEFINE_integer("train_steps", 100, "Number of training steps.")
flags.DEFINE_float("warmup_proportion", 0.1, "")


flags.DEFINE_bool("do_eval", False, "Evaluate the model.")
flags.DEFINE_string("eval_file", "dataset/val_128.tfrecords", "The validation file output by dataset.py.")
flags.DEFINE_integer("eval_steps", 3811//64, "Number of validation steps.")


flags.DEFINE_bool("do_predict", False, "Make prediction.")
flags.DEFINE_string("predict_file", "assets/icwb2-data/testing/pku_test.utf8", "The input file to be tokenized.")
flags.DEFINE_string("predict_output", "pku_pred.utf8", "The output file for tokenized result.")
flags.DEFINE_string("predict_model", "gs://berserker/export/1547563491", "The output file for tokenized result.")


flags.DEFINE_bool("do_export", False, "Export model.")
flags.DEFINE_string("output_dir", 'gs://berserker/export', "Output path for exported model.")


flags.DEFINE_bool("use_tpu", False, "Use TPU for training.")
flags.DEFINE_integer("num_tpu_cores", 8, "The number of TPU cores.")
flags.DEFINE_string("tpu_name", None, "TPU worker.")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf.gfile.MakeDirs(FLAGS.output_dir)

    model_fn = model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(
            os.path.join(FLAGS.gs_bert_model_ch_dir, 'bert_config.json')
        ),
        init_checkpoint=os.path.join(FLAGS.gs_bert_model_ch_dir, 'bert_model.ckpt'),
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=True if FLAGS.use_tpu else False,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.train_steps,
        num_warmup_steps=int(FLAGS.train_steps * FLAGS.warmup_proportion)
    )

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name) if FLAGS.use_tpu else None,
        model_dir=FLAGS.checkpoint_dir,
        save_checkpoints_steps=1000,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        )
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.batch_size
    )

    tf.logging.info('Setup success...')

    if FLAGS.do_train:
        estimator.train(
            input_fn=input_fn_builder(
                input_file=FLAGS.train_file,
                seq_length=FLAGS.max_seq_length,
                shuffle=True,
                repeat=True,
                drop_remainder=FLAGS.use_tpu
            ),
            steps=FLAGS.train_steps,
        )

    if FLAGS.do_eval:
        estimator.evaluate(
            input_fn=input_fn_builder(
                input_file=FLAGS.eval_file,
                seq_length=FLAGS.max_seq_length,
                shuffle=False,
                repeat=False,
                drop_remainder=FLAGS.use_tpu
            ),
            steps=FLAGS.eval_steps,
        )


    if FLAGS.do_export:
        estimator._export_to_tpu = False
        estimator.export_savedmodel(
            export_dir_base=FLAGS.output_dir,
            serving_input_receiver_fn=serving_input_fn_builder(
                seq_length=FLAGS.max_seq_length
            ),
            strip_default_attrs=True
        )

    if FLAGS.do_predict:
        import pandas as pd
        import numpy as np
        texts = pd.read_csv(FLAGS.predict_file, header=None, sep='^', skip_blank_lines=False)[0].fillna('')

        bert_inputs, mappings, sizes = batch_preprocess(
            texts,
            FLAGS.max_seq_length,
            FLAGS.batch_size
        )

        bert_outputs = estimator.predict(
            input_fn=predict_input_fn_builder(
                bert_inputs=bert_inputs,
                max_seq_length=FLAGS.max_seq_length,
                drop_remainder=FLAGS.use_tpu
            )
        )
        bert_outputs = [bert_output for bert_output in bert_outputs]

        for threshold in np.linspace(0.1, 0.9, 9):
            results = batch_postprocessing(
                texts,
                mappings,
                sizes,
                bert_inputs,
                bert_outputs,
                FLAGS.max_seq_length,
                threshold
            )
            assert len(results) == len(texts)
            with open('%.1f_'%threshold + FLAGS.predict_output, 'w') as f:
                for result in results:
                    print('  '.join(result), file=f)


if __name__ == "__main__":
  tf.app.run()
