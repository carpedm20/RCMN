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
               decay_step=10000, dataset="ptb", rnn_type="GRU", mode=0,
               l2=0.0004, optim="adam", is_single_output=False):
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
    self.is_single_output = is_single_output

    self.l2 = l2
    self.optim = optim
    self.batch_size = batch_size
    self.max_epoch = max_epoch
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.decay_rate = decay_rate
    self.decay_step = decay_step
    self.dataset = dataset
    self.mode = mode

    self._attrs = ["keep_prob", "hidden_dim", "embed_dim", "vocab_size",
                   "num_layers", "num_ks", "k_widths", "num_steps", "max_seq_l"
                   "rnn_type", "is_single_output", "l2", "optim", "batch_size", "max_epoch",
                   "learning_rate", "max_grad_norm", "decay_rate", "decay_step", "dataset"]

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

      rnn_inputs = tf.transpose(tf.pack(inputs), [1, 0, 2])

    with tf.variable_scope("lstm"):
      input_size = int(rnn_inputs.get_shape()[-1])

      if self.rnn_type == "GRU":
        cell1 = rnn_cell.GRUCell(self.hidden_dim, input_size)
        cell2 = rnn_cell.GRUCell(self.hidden_dim, self.hidden_dim)
      elif self.rnn_type == "LSTM":
        cell1 = rnn_cell.LSTMCell(self.hidden_dim, input_size)
        cell2 = rnn_cell.LSTMCell(self.hidden_dim, self.hidden_dim)
      else:
        raise Exception(" [!] Unkown rnn cell type: %s" % self.rnn_type)

      #if self.is_training and self.keep_prob < 1:
      #  cell1 = rnn_cell.dropoutwrapper(
      #      cell1, output_keep_prob=self.keep_prob)
      #  cell2 = rnn_cell.dropoutwrapper(
      #      cell2, output_keep_prob=self.keep_prob)

      cell = rnn_cell.MultiRNNCell(
          [cell1] + [cell2] * (self.num_layers-1))

      self._initial_state = cell.zero_state(self.batch_size, tf.float32)

      # [self.batch_size, self.max_seq_l, self.hidden_dim]
      outputs, state = tf.nn.dynamic_rnn(
          cell, rnn_inputs, [self.num_steps]*self.batch_size, dtype=tf.float32, scope="RNN")

    with tf.variable_scope("output"):
      if self.is_single_output:
        self.y = tf.placeholder(tf.int32, [self.batch_size])

        # [self.batch_size, self.vocab_size]
        self.y_ = rnn_cell.linear(tf.unpack(outputs), self.vocab_size, True, scope="y")
        self.y_idx = tf.argmax(self.y, 1)
      else:
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_l, self.vocab_size])

        y_words = []
        for idx, output in enumerate(tf.unpack(outputs)):
          if idx > 0:
            tf.get_variable_scope().reuse_variables()
          y_words.append(rnn_cell.linear(tf.unpack(tf.expand_dims(output, 0)), self.vocab_size, True, scope="y"))

        # [self.batch_size, self.max_seq_l, self.vocab_size]
        self.y_ = tf.pack(y_words)
        self.y_idx = tf.argmax(self.y, 2)

    with tf.variable_scope("training"):
      self.loss = tf.nn.softmax_cross_entropy_with_logits(self.y_, self.y)
      tvars = tf.trainable_variables()

      if self.l2 > 0:
        self.loss_l2 = self.l2 * tf.reduce_sum([tf.nn.l2_loss(tvar) for tvar in tvars])
      else:
        self.loss_l2 = 0

      self.cost = tf.reduce_sum(loss) / self.batch_size
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                        self.max_grad_norm)

      if self.optim == "adam":
        optimizer = tf.train.AdamOptimizer(self.lr)
      elif self.optim == "ada":
        optimizer = tf.train.AdagradOptimizer(self.lr)

      self.optim = optimizer.apply_gradients(zip(grads, tvars))

  def train(self):
    #epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()

    import ipdb; ipdb.set_trace() 

  @property
  def is_training(self):
    return self.mode == 0
