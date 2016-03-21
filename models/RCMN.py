import numpy as np
import tensorflow as tf

from .base import BaseModel

class RCMN(BaseModel):
  """Recurrent Convolutional Memory Network."""
  def __init__(self, sess, reader,
               keep_prob, hidden_dim, embed_dim, vocab_size,
               batch_size, num_steps, max_epoch, learning_rate,
               max_grad_norm, decay_Rate, decay_step, dataset):
    """Initialize Recurrent Convolutional Memory Network."""
    self.keep_prob = keep_prob
    self.hidden_dim = hidden_dim
    self.embed_dim = embed_dim
    self.vocab_size = vocab_size

    self.batch_size = batch_size
    self.num_steps = num_steps
    self.max_epoch = max_epoch
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.decay_rate = decay_rate
    self.decay_step = decay_step
    self.dataset = dataset

    self._attrs = ["keep_prob", "hidden_dim", "embed_dim", "vocab_size",
                   "batch_size", "num_steps", "max_epoch", "learning_rate",
                   "max_grad_norm", "decay_rate", "decay_step", "dataset"]

    self.build_model()

  def build_model(self):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)

    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

  def train(self):
    pass
