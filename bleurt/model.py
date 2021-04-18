# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""BLEURT's Tensorflow ops."""
from bleurt import checkpoint as checkpoint_lib
from bleurt.lib import modeling
from bleurt.lib import optimization
import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf
from tf_slim import metrics
import pdb
import pandas as pd
import itertools
import os
import hashlib

flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS

# BLEURT flags.
flags.DEFINE_string("bleurt_checkpoint_name", "bert_custom",
                    "Name of the BLEURT export to be created.")

flags.DEFINE_string("init_bleurt_checkpoint", None,
                    "Existing BLEURT export to be fine-tuned.")

# BERT flags.
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

# Flags to control training setup.
flags.DEFINE_enum("export_metric", "group_pairwise_accuracy", ["correlation", "kendalltau", 'total_abs_group_bias', "group_pairwise_accuracy"],
                  "Metric to chose the best model in export functions.")

flags.DEFINE_integer("shuffle_buffer_size", 500,
                     "Size of buffer used to shuffle the examples.")

# Flags to contol optimization.
flags.DEFINE_enum("optimizer", "adam", ["adam", "sgd", "adagrad"],
                  "Which optimizer to use.")

flags.DEFINE_float("learning_rate", 1e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

# BLEURT model flags.
flags.DEFINE_integer("n_hidden_layers", 0,
                     "Number of fully connected/RNN layers before prediction.")

flags.DEFINE_integer("hidden_layers_width", 128, "Width of hidden layers.")

flags.DEFINE_float("dropout_rate", 0,
                   "Probability of dropout over BERT embedding.")

# "Group Loss" training: turn this and "group_batches" on...
flags.DEFINE_bool("group_mse", False,
                   "Calculate MSE after averaging group predictions.")

flags.DEFINE_float("group_mean_alpha", 0.0,
                   "labels = alpha * group_mean + (1-alpha) * labels ")

flags.DEFINE_bool("group_batches", False,
                   "Create batches within groups")

# create reverse dictionary - define global var, and find year & lp from hash
group_hash_dict = {}

# adding custom hash function - we need deterministic hashing.
# def hash_md5_16(value):
#   b = bytes(value, 'utf-8')
#   w = hashlib.md5(b).hexdigest()[:16]
#   return int(w, 16)
def hash_md5_16(value):
  b = bytes(value, 'utf-8')
  w = hashlib.md5(b).hexdigest()[:16]
  val = int(w, 16)
  #convert to signed integer (to avoid overflow problems)
  bits = 64
  if val & (1 << (bits-1)):
    val -= 1 << bits
  return val


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, group_means, use_one_hot_embeddings, n_hidden_layers,
                 hidden_layers_width, dropout_rate):
  """Creates a regression model, loosely adapted from language/bert.

  Args:
    bert_config: `BertConfig` instance.
    is_training:  bool. true for training model, false for eval model.
    input_ids: int32 Tensor of shape [batch_size, seq_length].
    input_mask: int32 Tensor of shape [batch_size, seq_length].
    segment_ids: int32 Tensor of shape [batch_size, seq_length].
    labels: float32 Tensor of shape [batch_size].
    use_one_hot_embeddings:  Whether to use one-hot word embeddings or
      tf.embedding_lookup() for the word embeddings.
    n_hidden_layers: number of FC layers before prediction.
    hidden_layers_width: width of FC layers.
    dropout_rate: probability of dropout over BERT embedding.

  Returns:
    loss: <float32>[]
    per_example_loss: <float32>[batch_size]
    pred: <float32>[batch_size]
  """
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # <float>[batch_size, hidden_size]
  output_layer = model.get_pooled_output()
  bert_embed_size = output_layer.shape[-1]
  logging.info("BERT embedding width: {}".format(str(bert_embed_size)))
  if is_training and dropout_rate > 0:
    # Implements dropout on top of BERT's pooled output.
    # <float32>[batch_size, hidden_size]
    output_layer = tf.nn.dropout(output_layer, rate=dropout_rate)

  # Hidden layers
  for i in range(n_hidden_layers):
    # <float32>[batch_size, hidden_layers_width]
    logging.info("Adding hidden layer {}".format(i + 1))
    output_layer = tf.layers.dense(
        output_layer, hidden_layers_width, activation=tf.nn.relu)

  logging.info("Building linear output...")
  # <float32>[batch_size,1]
  predictions = tf.layers.dense(
      output_layer, 1, bias_initializer=tf.constant_initializer(0.15))
  # <float32>[batch_size]
  predictions = tf.squeeze(predictions, 1)
  # <float32>[batch_size]

  per_example_loss = tf.pow(predictions - labels, 2)

  if FLAGS.group_mean_alpha == 0.0:
      labels = labels
  elif FLAGS.group_mean_alpha == 1.0:
      labels = group_means
  else:
      alpha = FLAGS.group_mean_alpha
      labels = alpha * group_means + (1-alpha) * labels
      #labels = tf.Print(labels, [labels], 'labels: ')

  if not FLAGS.group_mse:
      loss = tf.reduce_mean(per_example_loss, axis=-1)
  else:
      pred_group_mean = tf.reduce_mean(predictions, axis=-1)
      true_group_mean = tf.reduce_mean(labels, axis=-1)
      loss = tf.pow(pred_group_mean - true_group_mean, 2)

  # <float32> []

  return (loss, per_example_loss, predictions)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, n_hidden_layers,
                     hidden_layers_width, dropout_rate):
  """Returns `model_fn` closure."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator/TPUEstimator."""
    logging.info("*** Building Regression BERT Model ***")
    tf.set_random_seed(55555)

    logging.info("*** Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s", name, features[name].shape)

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    group = features['group']
    group_mean = features['group_mean']

    if mode != tf.estimator.ModeKeys.PREDICT:
      scores = features["score"]
    else:
      scores = tf.zeros(tf.shape(input_ids)[0])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    total_loss, per_example_loss, pred = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, scores,
        group_mean, use_one_hot_embeddings, n_hidden_layers, hidden_layers_width,
        dropout_rate)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      # Loads pretrained model
      logging.info("**** Initializing from {} ****".format(init_checkpoint))
      tvars = tf.trainable_variables()
      initialized_variable_names = {}
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

      logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                     init_string)

      train_op = optimization.create_optimizer(total_loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu)

      if use_tpu:
        output_spec = tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)

      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:

      if use_tpu:
        eval_metrics = (metric_fn, [per_example_loss, pred, scores])
        output_spec = tf.estimator.TPUEstimatorSpec(
            mode=mode, loss=total_loss, eval_metric=eval_metrics)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=metric_fn(per_example_loss, pred, scores, group, group_mean))

    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode, predictions={"predictions": pred})

    return output_spec

  return model_fn


# TF ops to compute the metrics.
def concat_tensors(predictions, ratings, sources=None):
  """Concatenates batches of ratings and predictions."""
  concat_predictions_value, concat_predictions_update = \
      metrics.streaming_concat(predictions)
  concat_labels_value, concat_labels_update = \
      metrics.streaming_concat(ratings)
  if sources is None:
    return concat_predictions_value, concat_labels_value, \
        tf.group(concat_predictions_update, concat_labels_update)

  concat_sources_value, concat_sources_update = \
      metrics.streaming_concat(sources)
  return concat_predictions_value, concat_labels_value, concat_sources_value, \
        tf.group(concat_predictions_update, concat_labels_update,
                 concat_sources_update)


def kendall_tau_metric(predictions, ratings, weights=None):
  """Builds the computation graph for Kendall Tau metric."""

  def _kendall_tau(x, y):
    tau = stats.kendalltau(x, y)[0]
    return np.array(tau).astype(np.float32)

  if weights is not None:
    predictions = tf.boolean_mask(predictions, weights)
    ratings = tf.boolean_mask(ratings, weights)

  with tf.variable_scope("kendall_tau"):
    concat_predictions_value, concat_labels_value, update_op = (
        concat_tensors(predictions, ratings))
    metric_value = tf.reshape(
        tf.numpy_function(_kendall_tau,
                          [concat_predictions_value, concat_labels_value],
                          tf.float32),
        shape=[])

    return metric_value, update_op

def total_abs_group_bias(predictions, ratings, group):
  """Builds the computation graph for Kendall Tau metric."""

  def abs_group_bias(x, y, group):
    def to_components(index):
        return np.split(np.argsort(index), np.cumsum(np.unique(index, return_counts=True)[1]))

    # last array is empty for some reason
    split_groups = to_components(group)

    total = 0.
    for group_idx in split_groups[:-1]:
        group_x, group_y = x[group_idx], y[group_idx]
        total += np.abs(np.mean(group_x) - np.mean(group_y))

    total = -total      # export on best, lower bias is better
    return np.array(total).astype(np.float32)

  with tf.variable_scope("total_abs_group_bias"):
    concat_predictions_value, concat_labels_value, concat_groups_value, update_op = (
        concat_tensors(predictions, ratings, group))
    metric_value = tf.reshape(
        tf.numpy_function(abs_group_bias,
                          [concat_predictions_value, concat_labels_value, concat_groups_value],
                          tf.float32),
        shape=[])

    return metric_value, update_op


# def group_prmse(predictions, ratings, group):
#   """Builds the computation graph for Kendall Tau metric."""

#   def prmse(predictions, ratings, group):
#     def to_components(index):
#         return np.split(np.argsort(index), np.cumsum(np.unique(index, return_counts=True)[1]))

#     # last array is empty for some reason
#     split_groups = to_components(group)

#     total = 0.
#     for group_idx in split_groups[:-1]:
#       group_x, group_y = predictions[group_idx], ratings[group_idx]
#       se = (np.mean(group_x) - np.mean(group_y))**2
#       adjusted_se = se * len(group_x) - len(predictions)*np.var(ratings)
#       total += adjusted_se

#     total /= (len(split_groups)-1)
#     return np.array(total).astype(np.float32)

#   with tf.variable_scope("group_prmse"):
#     concat_predictions_value, concat_labels_value, concat_groups_value, update_op = (
#         concat_tensors(predictions, ratings, group))
#     metric_value = tf.reshape(
#         tf.numpy_function(prmse,
#                           [concat_predictions_value, concat_labels_value, concat_groups_value],
#                           tf.float32),
#         shape=[])

#     return metric_value, update_op


# assume that "group" is the proper group.
def group_pairwise_accuracy(predictions, ratings, group):
  """Builds the computation graph for Kendall Tau metric."""

  # this sorts the results by group, and then splits them by model/year/lp.
  # what are the "predictions" and "ratings" objects???
  def pairwise_accuracy(predictions, ratings, group):
    def to_components(index):
      return np.split(np.argsort(index), np.cumsum(np.unique(index, return_counts=True)[1]))

    split_groups = to_components(group)  # returns an array of arrays of indices (indices --> ratings array)

    grouped_scores = pd.DataFrame(columns=['bleurt_score', 'score', 'year_lp', 'group_name'])
    for group_idx in split_groups[:-1]:
      # last array is empty
      group_x, group_y = predictions[group_idx], ratings[group_idx]

      # save the group name + year/lp
      group_hash = group[group_idx[0]]
      year_lp = group_hash_dict[group_hash]

      # store the following [mean_bleurt_pred, mean_rating, year_lp, group]
      group_info = pd.DataFrame({'bleurt_score':[np.mean(group_x)], 'score':[np.mean(group_y)], 
        'year_lp':[year_lp], 'group_name':[group_hash]})
      grouped_scores = grouped_scores.append(group_info)

      #debug the above series creation
      # logging.info("attempting to add row to 'grouped_scores'...\n")
      # logging.info(f"mean prediction: {np.mean(group_x)}, year/lp: {year_lp}")
      # logging.info(str(group_info))

    #debug
    logging.info("Identified year, lp for all data points...\n")
    logging.info(str(grouped_scores.head()))

    # next, perform the pairwise score computation.
    total_pairs = 0
    bleurt_accuracy = 0.

    for i, g in grouped_scores.groupby('year_lp'):
      for (_, row), (_, row_) in itertools.combinations(g.iterrows(), r=2):
        total_pairs += 1
        if np.sign(row['bleurt_score'] - row_['bleurt_score']) == np.sign(row['score'] - row_['score']):
          bleurt_accuracy += 1

    accuracy = bleurt_accuracy / total_pairs
    #debug
    logging.info(
      "Pairwise accuracy computed. Total language pairs evaluated: {}, total correctly assessed: {}".format(
          str(total_pairs), str(bleurt_accuracy)))
    
    return np.array(accuracy).astype(np.float32)

  # what is this???
  with tf.variable_scope("group_pairwise_accuracy"):
    concat_predictions_value, concat_labels_value, concat_groups_value, update_op = (
        concat_tensors(predictions, ratings, group))
    metric_value = tf.reshape(
        tf.numpy_function(pairwise_accuracy,
                          [concat_predictions_value, concat_labels_value, concat_groups_value],
                          tf.float32),
        shape=[])

    return metric_value, update_op


def metric_fn(per_example_loss, pred, ratings, group, group_mean):
  """Metrics for BLEURT experiments."""
  # Mean of predictions
  mean_pred = tf.metrics.mean(values=pred)
  # Standard deviation of predictions
  mean = tf.reduce_mean(pred)
  diffs = tf.sqrt(tf.pow(pred - mean, 2))
  pred_sd = tf.metrics.mean(values=diffs)
  # Average squared error
  mean_loss = tf.metrics.mean(values=per_example_loss)
  # Average absolute error
  squared_diff = tf.pow(pred - ratings, 2)
  per_example_err = tf.sqrt(squared_diff)
  mean_err = tf.metrics.mean(per_example_err)
  # Pearson correlation
  correlation = metrics.streaming_pearson_correlation(pred, ratings)
  # Kendall Tau
  kendalltau = kendall_tau_metric(pred, ratings)

  # batched loss - TODO
  group_bias = total_abs_group_bias(pred, group_mean, group)

  # group prmse
  #group_prmse_value = group_prmse(pred, group_mean, group)

  # pairwise group loss
  group_pairwise_acc = group_pairwise_accuracy(pred, group_mean, group)

  output = {
      "eval_loss": mean_loss,
      "eval_mean_err": mean_err,
      "eval_mean_pred": mean_pred,
      "eval_pred_sd": pred_sd,
      "correlation": correlation,
      "kendalltau": kendalltau,
      "total_abs_group_bias": group_bias,
      #"group_prmse": group_prmse_value,
      "group_pairwise_accuracy": group_pairwise_acc
  }

  return output


def input_fn_builder(tfrecord_file,
                     seq_length,
                     is_training,
                     batch_size,
                     drop_remainder=True):
  """Creates an `input_fn` closure to be passed to Estimator."""
  logging.info(
      "Creating input fun with batch_size: {} and drop remainder: {}".format(
          str(batch_size), str(drop_remainder)))
  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "score": tf.FixedLenFeature([], tf.float32),
      "group": tf.FixedLenFeature([], tf.int64),
      "group_mean": tf.FixedLenFeature([], tf.float32),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.

    # I don't have a TPU
    TPU = False
    for name in list(example.keys()):
      t = example[name]
      if TPU and t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t
    return example

  def input_fn(params):  # pylint: disable=unused-argument
    """Acutal data generator."""
    d = tf.data.TFRecordDataset(tfrecord_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=FLAGS.shuffle_buffer_size)

    d = d.map(lambda record: _decode_record(record, name_to_features))

    if FLAGS.group_batches:
      logging.info("Group batches activated, each batch will have examples within the group.")
      d = d.apply(tf.data.experimental.group_by_window(
        key_func=lambda elem: elem['group'],
        reduce_func=lambda _, window: window.batch(batch_size=batch_size, drop_remainder=drop_remainder),
        window_size=batch_size))
    else:
      d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def _model_comparator(best_eval_result, current_eval_result):
  metric = FLAGS.export_metric
  return best_eval_result[metric] <= current_eval_result[metric]


def _serving_input_fn_builder(seq_length):
  """Input function for exported models."""
  # We had to use `tf.zeros` instead of the usual
  # `tf.placeholder(tf.int64, shape=[None, seq_length])` to be compatible with
  # TF2's eager mode, which deprecates all calls to `tf.placeholder`.
  if tf.executing_eagerly():
    name_to_features = {
        "input_ids": tf.zeros(dtype=tf.int64, shape=[0, seq_length]),
        "input_mask": tf.zeros(dtype=tf.int64, shape=[0, seq_length]),
        "segment_ids": tf.zeros(dtype=tf.int64, shape=[0, seq_length]),
        "group": tf.zeros(dtype=tf.int64, shape=[0,]),
        "group_mean": tf.zeros(dtype=tf.float32, shape=[0,]),
    }
  else:
    name_to_features = {
        "input_ids": tf.placeholder(tf.int64, shape=[None, seq_length]),
        "input_mask": tf.placeholder(tf.int64, shape=[None, seq_length]),
        "segment_ids": tf.placeholder(tf.int64, shape=[None, seq_length]),
        "group": tf.placeholder(tf.int64, shape=[None,]),
        "group_mean": tf.placeholder(tf.float32, shape=[None,]),
    }
  return tf.estimator.export.build_raw_serving_input_receiver_fn(
      name_to_features)


def run_finetuning(train_set,
                   dev_set,
                   scratch_dir,
                   train_tfrecord,
                   dev_tfrecord,
                   train_eval_fun=None,
                   use_tpu=False,
                   additional_train_params=None):
  """Main function to train and eval BLEURT."""

  logging.info("Initializing BLEURT training pipeline.")

  bleurt_params = checkpoint_lib.get_bleurt_params_from_flags_or_ckpt()
  max_seq_length = bleurt_params["max_seq_length"]
  bert_config_file = bleurt_params["bert_config_file"]
  init_checkpoint = bleurt_params["init_checkpoint"]

  logging.info("Creating input data pipeline.")
  logging.info("Train/Eval batch size: {}".format(str(FLAGS.batch_size)))

  # set up the training "reverse-dictionary" to capture year-lp
  logging.info("Starting to populate reverse group dictionary.")
  train_df = pd.read_json(train_set, lines=True)
  dev_df = pd.read_json(dev_set, lines=True)
  examples_df = pd.concat([train_df, dev_df])
  #group_hash_dict = {}
  for g in examples_df['group'].unique():
    h = hash_md5_16(g)
    year_lp = '|'.join(g.split('|')[1:])
    group_hash_dict[h] = year_lp
  
    # debugging
    logging.info(f"Example - {g}:{h}:{group_hash_dict[h]}\n")
  logging.info("Group hash dict populated!")

  #   == also, save the dictionary to a file for debugging purposes
  # with open(os.path.join(scratch_dir, 'group_hash_dict'), 'w') as f:
  #   f.write(str(group_hash_dict)+'\n') 

  train_input_fn = input_fn_builder(
      train_tfrecord,
      seq_length=max_seq_length,
      is_training=True,
      batch_size=FLAGS.batch_size,
      drop_remainder=use_tpu)

  dev_input_fn = input_fn_builder(
      dev_tfrecord,
      seq_length=max_seq_length,
      is_training=False,
      batch_size=FLAGS.batch_size,
      drop_remainder=use_tpu)

  logging.info("Creating model.")
  bert_config = modeling.BertConfig.from_json_file(bert_config_file)
  num_train_steps = FLAGS.num_train_steps
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_tpu,
      n_hidden_layers=FLAGS.n_hidden_layers,
      hidden_layers_width=FLAGS.hidden_layers_width,
      dropout_rate=FLAGS.dropout_rate)

  logging.info("Creating TF Estimator.")
  exporters = [
      tf.estimator.BestExporter(
          "bleurt_best",
          serving_input_receiver_fn=_serving_input_fn_builder(max_seq_length),
          event_file_pattern="eval_default/*.tfevents.*",
          compare_fn=_model_comparator,
          exports_to_keep=1)
  ]
  tf.enable_resource_variables()

  logging.info("*** Entering the Training / Eval phase ***")
  if not additional_train_params:
    additional_train_params = {}
  train_eval_fun(
      model_fn=model_fn,
      train_input_fn=train_input_fn,
      eval_input_fn=dev_input_fn,
      exporters=exporters,
      **additional_train_params)
