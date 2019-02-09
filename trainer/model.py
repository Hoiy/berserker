from trainer.ext import modeling
from trainer.ext import optimization
import tensorflow as tf

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()
  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]

  logits = tf.reshape(tf.keras.layers.Dense(1)(final_hidden), (batch_size, seq_length))
  predictions = tf.nn.sigmoid(logits)

  return predictions


def model_fn_builder(bert_config, init_checkpoint, use_tpu, use_one_hot_embeddings,
                     learning_rate=None, num_train_steps=None, num_warmup_steps=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    truths = features["truths"]

    predictions = create_model(
        bert_config=bert_config,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}

    #############
    # scaffold_fn
    #############
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    ############
    # total_loss
    ############
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        input_mask_f = tf.cast(input_mask, tf.float32)
        per_example_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(truths, predictions) * input_mask_f, axis=-1) / tf.reduce_sum(input_mask_f, axis=-1)
        total_loss = tf.reduce_mean(per_example_loss)


    #############
    # output_spec
    #############
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      assert learning_rate is not None
      assert num_train_steps is not None
      assert num_warmup_steps is not None
      train_op = optimization.create_optimizer(
          total_loss,
          learning_rate,
          num_train_steps,
          num_warmup_steps,
          use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(truths, predictions, masks):
        threshold = 0.5
        pred_int = tf.cast(predictions >= threshold, tf.int32)
        return {
            "auc": tf.metrics.auc(
                labels=truths,
                predictions=predictions,
                weights=masks
            ),
            "f1_score @%.2f"%threshold: tf.contrib.metrics.f1_score(
                labels=truths,
                predictions=pred_int,
                weights=masks
            ),
            "precision @%.2f"%threshold: tf.metrics.precision(
                labels=truths,
                predictions=pred_int,
                weights=masks
            ),
            "recall @%.2f"%threshold: tf.metrics.recall(
                labels=truths,
                predictions=pred_int,
                weights=masks
            ),
        }

      eval_metrics = (metric_fn, [truths, predictions, input_mask])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
            "predictions": predictions
          },
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn
