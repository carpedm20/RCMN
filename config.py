
from utils import pp, class_vars

class Char(object):
  word = 0

class Word(object):
  word = 1

class LSTM(object):
  rnn_type = "LSTM" # Much better than GRU

class GRU(object):
  rnn_type = "GRU"

class Default(object):
  dataset = "ptb"
  is_single_output = True
  max_pool_in_output = True

class RcmnSmall(object):
  keep_prob     = 1.0
  embed_dim     = 150
  hidden_dim    = 150
  vocab_size    = 10000
  word          = 1
  num_steps     = 2
  max_seq_l     = 30
  num_layers    = 2
  k_widths      = [3]
  num_ks        = [4]

  batch_size    = 20

class RcmnMedium(object):
  keep_prob     = 0.5
  embed_dim     = 350
  hidden_dim    = 4500
  vocab_size    = 10000
  word          = 1
  num_steps     = 5
  max_seq_l     = 25
  num_layers    = 1
  k_widths      = [2]
  num_ks        = [10]

  batch_size    = 30

class RcmnLarge(object):
  keep_prob     = 0.35
  embed_dim     = 500
  hidden_dim    = 6500
  vocab_size    = 10000
  word          = 1
  num_steps     = 10
  max_seq_l     = 30
  num_layers    = 4
  k_widths      = [2]
  num_ks        = [10]

  batch_size    = 30

class RcmnTraining1(object):
  max_epoch     = 1000
  max_grad_norm = 10
  decay_rate    = 0.96
  decay_step    = 10000
  learning_rate = 0.001
  l2            = 0.0004
  epsilon       = 0.1
  optim_type    = "adam"

class RcmnSmallConfig(RcmnSmall, Default, Word, LSTM, RcmnTraining1):
  pass

class RcmnLargeConfig(RcmnLarge, Default, Char, GRU, RcmnTraining1):
  pass

def get_config(FLAGS):
  if FLAGS.model == "rcmn":
    config = FLAGS
  elif FLAGS.model == "small":
    config = RcmnSmallConfig
  elif FLAGS.model == "large":
    config = RcmnLargeConfig
  else:
    raise ValueError(" [!] Invalid model: %s", FLAGS.model)

  if FLAGS.model == "ccmn":
    pp(FLAGS.__flags)
  else:
    pp(class_vars(config))

  return config
