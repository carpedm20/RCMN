import os
import tensorflow as tf

from utils import pp
from models import RCMN
from config import *

from tensorflow.models.rnn.ptb import reader as ptb_reader

flags = tf.app.flags

# Model configs
flags.DEFINE_string("model", "rcmn", "The name of model [rcmn]")
flags.DEFINE_string("rnn_type", "GRU", "The type of rnn cell [GRU]")
flags.DEFINE_float("keep_prob", 0.35, "Probability on dropout [0.35]")
flags.DEFINE_float("num_layers", 2, "The number of layers in RCMN [2]")
flags.DEFINE_integer("hidden_dim", 650, "The dimension of hidden memory [650]")
flags.DEFINE_integer("embed_dim", 500, "The dimension of input embeddings [500]")
flags.DEFINE_integer("vocab_size", 10000, "The size of vocabulary [10000]")
flags.DEFINE_integer("mode", 0, "0 for training, 1 for testing [0]")
flags.DEFINE_integer("word", 1, "0 for using character, 1 for using word [1]")

# Training configs
flags.DEFINE_integer("max_epoch", 14, "Maximum number of epoch [14]")
flags.DEFINE_integer("num_steps", 50, "Maximum number of epoch [50]")
flags.DEFINE_integer("batch_size", 20, "Maximum number of epoch [14]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of optimizer [0.001]")
flags.DEFINE_float("max_grad_norm", 10, "Maximum gradient normalization term [10]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_string("dataset", "ptb", "The name of dataset [ptb]")

FLAGS = flags.FLAGS


def main(_):
  with tf.Session() as sess:
    config = get_config(FLAGS) or FLAGS

    model = RCMN(sess, dataset=config.dataset, rnn_type=config.rnn_type,
                 keep_prob=config.keep_prob, hidden_dim=config.hidden_dim,
                 num_layers=config.num_layers, embed_dim=config.embed_dim,
                 vocab_size=config.vocab_size, batch_size=config.batch_size,
                 num_steps=config.num_steps, max_epoch=config.max_epoch,
                 learning_rate=config.learning_rate, max_seq_l=config.max_seq_l,
                 k_widths=config.k_widths, num_ks=config.num_ks,
                 max_grad_norm=config.max_grad_norm, decay_rate=config.decay_rate,
                 decay_step=config.decay_step, is_single_output=config.is_single_output,
                 max_pool_in_output=config.max_pool_in_output, epsilon=config.epsilon,
                 l2=config.l2, optim_type=config.optim_type, mode=FLAGS.mode)

    if FLAGS.mode == 0:
      model.train()
    else:
      model.test()

if __name__ == '__main__':
  tf.app.run()
