from utils import pp, class_vars

class Char(object):
  word = 0

class Word(object):
  word = 1

class LSTM(object):
  rnn_type = "LSTM"

class GRU(object):
  rnn_type = "GRU"

class Default(object):
  dataset = "ptb"

class RcmnSmall(object):
  keep_prob     = 1.0
  embed_dim     = 200
  hidden_dim    = 400
  vocab_size    = 10000
  word          = 1
  num_steps     = 3
  max_seq_l     = 20
  num_layers    = 2
  k_widths      = [2]
  num_ks        = [5]

  batch_size    = 10

class RcmnMedium(object):
  keep_prob     = 0.5
  embed_dim     = 350
  hidden_dim    = 450
  vocab_size    = 10000
  word          = 1
  num_steps     = 5
  max_seq_l     = 25
  num_layers    = 3
  k_widths      = [2]
  num_ks        = [10]

  batch_size    = 20

class RcmnLarge(object):
  keep_prob     = 0.35
  embed_dim     = 500
  hidden_dim    = 650
  vocab_size    = 10000
  word          = 1
  num_steps     = 10
  max_seq_l     = 30
  num_layers    = 4
  k_widths      = [2]
  num_ks        = [10]

  batch_size    = 30

class RcmnTraining1(object):
  max_epoch     = 14
  learning_rate = 0.001
  max_grad_norm = 10
  decay_rate    = 0.96
  decay_step    = 10000

class RcmnSmallConfig(RcmnSmall, Default, Word, GRU, RcmnTraining1):
  pass

class RcmnLargeConfig(RcmnLarge, Default, Char, GRU, RcmnTraining1):
  pass

def get_config(FLAGS):
  if FLAGS.model == "rcmn":
    config = FLAGS
  elif FLAGS.model == "rcmn_small":
    config = RcmnSmallConfig
  elif FLAGS.model == "rcmn_large":
    config = RcmnLargeConfig
  else:
    raise ValueError(" [!] Invalid model: %s", FLAGS.model)

  if FLAGS.model == "ccmn":
    pp(FLAGS.__flags)
  else:
    pp(class_vars(config))

  return config
