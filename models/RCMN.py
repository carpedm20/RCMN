import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn.ptb import reader as ptb_reader

from .ops import *
from .base import BaseModel

class RCMN(BaseModel):
  """Recurrent Convolutional Memory Network."""
  def __init__(self, sess, keep_prob=0.35, hidden_dim=650, num_layers=2,
               embed_dim=500, k_widths=[2], num_ks=[5], num_steps=5,
               vocab_size=10000, batch_size=20, max_seq_l=30, max_epoch=100,
               learning_rate=0.001, max_grad_norm=10, decay_rate=0.96,
               decay_step=10000, dataset="ptb", rnn_type="GRU", mode=0):
    """Initialize Recurrent Convolutional Memory Network."""
    self.keep_prob = keep_prob
    self.hidden_dim = hidden_dim
    self.embed_dim = embed_dim
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.num_ks = num_ks
    self.k_widths = k_widths
    self.num_steps = num_steps
    self.max_seq_l = max_seq_l
    self.rnn_type = rnn_type

    self.batch_size = batch_size
    self.max_epoch = max_epoch
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.decay_rate = decay_rate
    self.decay_step = decay_step
    self.dataset = dataset
    self.mode = mode

    self._attrs = ["keep_prob", "hidden_dim", "embed_dim", "vocab_size",
                   "batch_size", "max_seq_l", "max_epoch", "learning_rate",
                   "max_grad_norm", "decay_rate", "decay_step", "dataset"]

    self.build_reader()
    self.build_model()

  def build_reader(self):
    data_path = "./data/%s" % self.dataset

    if self.dataset == 'ptb':
      raw_data = ptb_reader.ptb_raw_data(data_path)
      self.train_data, self.valid_data, self.test_data, _ = raw_data
    else:
      raise ValueError(" [!] Unkown dataset: %s" % data_path)

  def build_model(self):
    with tf.variable_scope("conv"):
      self.x = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_l])

      with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [self.vocab_size, self.embed_dim])
        first_input = tf.nn.embedding_lookup(embedding, self.x)

      if self.is_training and self.keep_prob < 1:
        first_input = tf.nn.dropout(first_input, self.keep_prob)
      first_input = tf.expand_dims(first_input, -1)

      inputs = []
      for step in xrange(self.max_seq_l):
        for idx, (k_width, k_dim) in enumerate(zip(self.k_widths, self.num_ks)):
          name = "conv_s:%d_i:%d_w:%d_d:%d" % (step, idx, k_width, k_dim)
          if step == 0:
            k_width = 1
            k_dim = self.num_ks[0]
            conv = first_input

          conv = conv2d(conv, k_dim, k_width, self.embed_dim, name=name)
          inputs.append(tf.reshape(conv, [self.batch_size, -1]))

      inputs = tf.transpose(tf.pack(inputs), [1, 0, 2])

    with tf.variable_scope("lstm"):
      input_size = inputs.get_shape()[-1]

      if self.rnn_type == "GRU":
        cell1 = rnn_cell.GRUCell(self.hidden_dim, input_size)
        cell2 = rnn_cell.GRUCell(self.hidden_dim, self.hidden_dim)
      elif self.rnn_type == "LSTM":
        cell1 = rnn_cell.LSTMCell(self.hidden_dim, input_size, forget_bias=0.0)
        cell2 = rnn_cell.LSTMCell(self.hidden_dim, self.hidden_dim, forget_bias=0.0)
      else:
        raise Exception(" [!] Unkown rnn cell type: %s" % self.rnn_type)

      if self.is_training and self.keep_prob < 1:
        cell1 = tf.nn.rnn_cell.dropoutwrapper(
            cell1, output_keep_prob=self.keep_prob)
        cell2 = tf.nn.rnn_cell.dropoutwrapper(
            cell2, output_keep_prob=self.keep_prob)

      cell = tf.nn.rnn_cell.MultiRNNCell(
          [cell1] + [cell2] * (self.num_layers-1))

      self._initial_state = cell.zero_state(self.batch_size, tf.float32)

      outputs, state = tf.nn.dynamic_rnn(
          cell, inputs, [self.num_steps]*self.batch_size, dtype=tf.float32, scope="RNN")

      print outputs

    with tf.variable_scope("output"):
      self.y = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_l])

  def train(self):
    pass

  @property
  def is_training(self):
    return self.mode == 0
