import time
import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
import ptb_reader

from .ops import *
from .base import BaseModel

class RCMN(BaseModel):
  """Recurrent Convolutional Memory Network."""
  def __init__(self, sess,
               keep_prob,#=0.35,
               hidden_dim,#=650,
               embed_dim,#=500,
               num_layers,#=2,
               vocab_size,#=10000,
               num_ks,#=[5],
               k_widths,#=[2],
               num_steps,#=5,
               max_seq_l,#=30,
               rnn_type,#="GRU", 
               is_single_output,#=False,
               max_pool_in_output,#=False,

               l2,#=0.0004,
               optim_type,#="adam",
               batch_size,#=20,
               max_epoch,#=100,
               learning_rate,#=0.001,
               max_grad_norm,#=10,
               epsilon,#=1.0
               decay_rate,#=0.96,
               decay_step,#=10000,
               dataset,#="ptb",
               mode,#=0,
               ):
               
    """Initialize Recurrent Convolutional Memory Network."""
    self.keep_prob = keep_prob
    self.hidden_dim = hidden_dim
    self.embed_dim = embed_dim
    self.num_layers = num_layers
    self.vocab_size = vocab_size
    self.num_ks = num_ks
    self.k_widths = k_widths
    self.num_steps = num_steps
    self.max_seq_l = max_seq_l
    self.rnn_type = rnn_type
    self.is_single_output = is_single_output
    self.max_pool_in_output = max_pool_in_output

    self.l2 = l2
    self.optim_type = optim_type
    self.batch_size = batch_size
    self.max_epoch = max_epoch
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.epsilon = epsilon
    self.decay_rate = decay_rate
    self.decay_step = decay_step
    self.dataset = dataset
    self.mode = mode

    self.sess = sess
    self.checkpoint_dir = "checkpoints"

    self.g_step = tf.Variable(0, name='step', trainable=False)
    self.g_epoch = tf.Variable(0, name='epoch', trainable=False)

    self._attrs = ["keep_prob", "hidden_dim", "num_layers",
                   "embed_dim", "k_widths", "num_ks", "num_steps",
                   "vocab_size", "batch_size", "max_seq_l", "max_epoch",
                   "learning_rate", "max_grad_norm", "decay_rate",
                   "decay_step", "dataset", "rnn_type",
                   "l2", "epsilon", "optim_type", "is_single_output",
                   "max_pool_in_output"]

    self.build_model()
    self.build_reader()

  def build_model(self):
    with tf.variable_scope("conv"):
      self.x = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_l])

      with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [self.vocab_size, self.embed_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
        first_input = tf.nn.embedding_lookup(embedding, self.x)

      if self.is_training and self.keep_prob < 1:
        first_input = tf.nn.dropout(first_input, self.keep_prob)
      first_input = tf.expand_dims(first_input, -1)

      inputs = []
      for step in xrange(self.num_steps):
        for idx, (k_width, k_dim) in enumerate(zip(self.k_widths, self.num_ks)):
          name = "conv_S%d_I%d_W%d_D%d" % (step, idx, k_width, k_dim)
          if step == 0:
            k_width = 1
            k_dim = self.num_ks[0]
            conv = first_input

          conv = tf.nn.relu(conv2d(conv, k_dim, k_width, self.embed_dim, name=name),
                            name=name.replace('conv', 'relu'))
          inputs.append(tf.reshape(conv, [self.batch_size, -1]))

          tf.contrib.layers.summarize_activation(conv)

    with tf.variable_scope("lstm"):
      input_size = int(inputs[0].get_shape()[-1])

      if self.rnn_type == "GRU":
        cell1 = rnn_cell.GRUCell(self.hidden_dim, input_size)
        cell2s = []
        for _ in xrange(self.num_layers-1):
          cell2s.append(rnn_cell.GRUCell(self.hidden_dim, self.hidden_dim))
      elif self.rnn_type == "LSTM":
        cell1 = rnn_cell.LSTMCell(self.hidden_dim, input_size)
        cell2s = []
        for _ in xrange(self.num_layers-1):
          cell2s.append(rnn_cell.LSTMCell(self.hidden_dim, self.hidden_dim))
      else:
        raise Exception(" [!] Unkown rnn cell type: %s" % self.rnn_type)

      #if self.is_training and self.keep_prob < 1:
      #  cell1 = rnn_cell.dropoutwrapper(
      #      cell1, output_keep_prob=self.keep_prob)
      #  cell2 = rnn_cell.dropoutwrapper(
      #      cell2, output_keep_prob=self.keep_prob)

      cell = rnn_cell.MultiRNNCell(
          [cell1] + cell2s)

      self.initial_state = cell.zero_state(self.batch_size, tf.float32)

      # Need to be fixed : https://github.com/tensorflow/tensorflow/issues/1306
      if False:
        rnn_inputs = tf.pack(inputs)

        # [self.max_seq_l, self.batch_size, self.hidden_dim]
        cell_output, state = tf.nn.dynamic_rnn(
            cell, rnn_inputs, [self.num_steps]*self.batch_size,
            initial_state=self.initial_state, time_major=True, scope="RNN")
      else:
        outputs = []
        state = self.initial_state

        with tf.variable_scope("RNN"):
          for time_step in range(self.num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[time_step], state)
            outputs.append(cell_output)

        cell_output = tf.pack(outputs)

    with tf.variable_scope("output"):
      if self.is_single_output:
        self.y = tf.placeholder(tf.int64, [self.batch_size], name="y")
        self.y_one_hot = tf.one_hot(self.y, self.vocab_size, 1.0, 0.0)

        # self.max_seq_l x [self.batch_size, self.hidden_dim]
        outputs = tf.unpack(cell_output)

        self.Y_ = []
        for step, output in enumerate(outputs):
          # [self.batch_size, self.vocab_size]
          logits = linear(output, self.vocab_size, True, scope="y_S%d" % step)
          self.Y_.append(tf.reshape(logits, [self.batch_size, -1]))
      else:
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_l], name="y")

        # self.max_seq_l x [self.batch_size, self.hidden_dim]
        outputs = tf.unpack(cell_output)

        self.Y_ = []
        for step, output in enumerate(outputs):
          # [self.batch_size x self.max_seq_l, self.vocab_size]
          logits = linear(output, self.max_seq_l * self.vocab_size, True, scope="y_S%d" % step)
          self.Y_.append(tf.reshape(logits, [self.batch_size * self.max_seq_l, -1]))

    with tf.variable_scope("training"):
      if self.is_single_output:
        if self.max_pool_in_output:
          self.y_ = tf.squeeze(tf.nn.max_pool(
            tf.expand_dims(tf.pack(self.Y_), [0]), [1, self.num_steps, 1, 1], [1, 1, 1, 1], 'VALID'))
        else:
          self.y_ = self.Y_[-1]

        self.pred = tf.argmax(self.y_, 1, name="predictions")
        seq_loss = tf.nn.softmax_cross_entropy_with_logits(self.y_, self.y_one_hot)
      else:
        if self.max_pool_in_output:
          self.y_ = tf.squeeze(tf.nn.max_pool(
            tf.expand_dims(tf.pack(self.Y_), 0), [1, 3, 1, 1], [1, 1, 1, 1], 'VALID'))
        else:
          self.y_ = self.Y_[-1]

        seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
            [self.y_],
            [tf.reshape(self.y, [-1])],
            [tf.ones([self.batch_size * self.max_seq_l])])

      tvars = tf.trainable_variables()
      if self.l2 > 0:
        self.l2_loss = sum([tf.contrib.layers.l2_regularizer(self.l2)(tvar) for tvar in tvars])
      else:
        self.l2_loss = 0

      loss = tf.reduce_mean(seq_loss)
      self.cost = loss + self.l2_loss
      self.final_state = state

      grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)

      if self.optim_type == "adam":
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
      elif self.optim_type == "ada":
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)

      self.optim = optimizer.apply_gradients(zip(grads, tvars), global_step=self.g_step)

      _ = tf.scalar_summary("perplexity", tf.exp(self.cost))
      _ = tf.scalar_summary("loss + l2", self.cost)
      _ = tf.scalar_summary("l2", self.l2_loss)
      _ = tf.scalar_summary("loss", loss)
      _ = tf.histogram_summary("loss hist", seq_loss)

  def build_reader(self):
    data_path = "./data/%s" % self.dataset

    if self.dataset == 'ptb':
      self.reader = ptb_reader

      raw_data = self.reader.ptb_raw_data(data_path)
      self.train_data, self.valid_data, self.test_data, self.word2idx = raw_data
      self.idx2word = {i:w for w, i in self.word2idx.items()}

      self.iterator = self.reader.ptb_iterator
    else:
      raise ValueError(" [!] Unkown dataset: %s" % data_path)

  def train(self):
    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/%s" % self.get_model_dir(), self.sess.graph)

    tf.initialize_all_variables().run()
    self.load_model()

    start_epoch = self.g_epoch.eval()
    for epoch in xrange(start_epoch, self.max_epoch-start_epoch):
      train_perplexity = self.run_epoch(self.train_data, merged_sum, writer)
      print(" [*] Epoch: %d Train Perplexity: %.3f" % (epoch + 1, train_perplexity))

      test_perplexity = self.run_epoch(self.test_data)
      print(" [*] Epoch: %d Test Perplexity: %.3f" % (epoch + 1, test_perplexity))

      summary = tf.Summary(value=[tf.Summary.Value(tag="test_perplexity", simple_value=test_perplexity)])
      writer.add_summary(summary, global_step=epoch)

      self.g_epoch.assign(epoch + 1).eval()
      self.save_model(step=self.g_step.eval())

  def test(self):
    print("[*] Start testing...")

    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/%s" % self.get_model_dir(), self.sess.graph)

    tf.initialize_all_variables().run()
    self.load_model()

    #train_perplexity = self.run_epoch(self.train_data, merged_sum, writer)
    #print(" [*] Train Perplexity: %.3f" % (train_perplexity))

    test_perplexity = self.run_epoch(self.test_data)
    print(" [*] Test Perplexity: %.3f" % (test_perplexity))

  def run_epoch(self, data, summary=None, writer=None):
    epoch_size = (len(data) // self.batch_size) - self.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = self.initial_state.eval()

    iterator = self.iterator(data, self.batch_size, self.max_seq_l)
    for step, (x, y) in enumerate(iterator):
      if self.is_single_output:
        data = {self.x: x, self.y: y[:,-1], self.initial_state: state}
      else:
        data = {self.x: x, self.y: y, self.initial_state: state}

      if step % 100 == 99:
        cost, state = self.sess.run([self.cost, self.final_state], data)
        print(" [*] %.3f perplexity: %.3f speed: %.0f wps" %
              (step * 1.0 / epoch_size, np.exp(costs / iters),
              iters * self.batch_size / (time.time() - start_time)))

      if summary != None:
        if step % 100 == 99:
          cost, state, summary_str, _ = self.sess.run(
              [self.cost, self.final_state, summary, self.optim], data)
          writer.add_summary(summary_str, self.g_step.eval())
        else:
          cost, state, _ = self.sess.run([self.cost, self.final_state, self.optim], data)
      else:
        cost, state = self.sess.run([self.cost, self.final_state], data)

      costs += cost
      if self.is_single_output:
        iters += 1
      else:
        iters += self.num_steps

    pred_words = [self.idx2word[i] for i in self.sess.run(self.pred, data)]
    print("\n\n".join([" ".join([self.idx2word[i] for i in l]) \
                      + ' "%s:%s"' % (w, self.idx2word[y_]) for l, w, y_ in zip(x, pred_words, y[:,-1])]))

    return np.exp(costs / iters)

  @property
  def is_training(self):
    return self.mode == 0
