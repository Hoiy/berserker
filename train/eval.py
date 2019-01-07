import os
import tensorflow as tf
import pandas as pd
from train.config import TRAINING_DATA_PATH, TESTING_DATA_PATH
from train.transform import create_tokenizer, text_to_tfexample
from train.input import input_fn_builder
from train.model import model_fn_builder
from tqdm import tqdm
from assets.bert import modeling

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir", None,
    "The directory where the model checkpoints are written.")

# flags.DEFINE_string(
#     "bert_model_dir", None,
#     "The directory where the bert model checkpoints are written.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
flags.DEFINE_integer("predict_batch_size", 64, "Prediction batch size.")
# flags.DEFINE_integer("save", 64, "Prediction batch size.")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    input_fn = input_fn_builder(
        input_file='gs://dev-test-bert-tpu/bert-chinese-tokenizer/dataset/test.tfrecords',
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=True
    )

    model_fn = model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(
            os.path.join('assets', 'chinese_L-12_H-768_A-12', 'bert_config.json')
        ),
        init_checkpoint=os.path.join('assets', 'chinese_L-12_H-768_A-12', 'bert_model.ckpt'),
        learning_rate=None,
        num_train_steps=None,
        num_warmup_steps=None,
        use_tpu=False,
        use_one_hot_embeddings=True
    )

    # tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver('localhost')
    # run_config = tf.contrib.tpu.RunConfig()
        # cluster=tpu_cluster_resolver,
        # model_dir=FLAGS.model_dir,
        # save_checkpoints_steps=None,
        # tpu_config=None
    # )
    # tf.contrib.tpu.TPUConfig(
    #         iterations_per_loop=ITERATIONS_PER_LOOP,
    #         num_shards=NUM_TPU_CORES,
    #         per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    # # run_config,
    # model_fn

    estimator = tf.estimator.Estimator(model_fn=model_fn, params={
        # 'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        # 'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'batch_size': FLAGS.predict_batch_size
    }, model_dir=FLAGS.model_dir)

    # estimator = tf.contrib.tpu.TPUEstimator(
    #     use_tpu=False,
    #     model_fn=model_fn,
    #     config=run_config,
    #     predict_batch_size=FLAGS.predict_batch_size
    # )

    test_pred_gen = estimator.predict(input_fn=input_fn)
    print(next(test_pred_gen))


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  tf.app.run()




# SELECT article_id, count(1) FROM (
# SELECT DISTINCT source_id AS article_id, FROM_UNIXTIME(score) AS publish_start_date
#                 FROM hk01_newsfeed.feed_item
#                 WHERE type = 1
#                 AND source_category = 0
#                 AND score < UNIX_TIMESTAMP()
#                 AND source_id NOT IN (143167, 151065)
#                 ORDER BY score DESC
#                 LIMIT 1000
# ) t
# group by article_id
# order by count(1) desc
