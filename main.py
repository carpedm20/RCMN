import os
import tensorflow as tf

from utils import pp
from models import RCMN
from tensorflow.models.rnn.ptb import reader

flags = tf.app.flags

# Model configs
flags.DEFINE_string("model", "RCMN", "The name of model [rcmn]")
flags.DEFINE_float("keep_prob", 0.35, "Probability on dropout [0.35]")
flags.DEFINE_integer("hidden_dim", 650, "The dimension of hidden memory [650]")
flags.DEFINE_integer("embed_dim", 500, "The dimension of input embeddings [500]")
flags.DEFINE_integer("vocab_size", 10000, "The size of vocabulary [10000]")
flags.DEFINE_boolean("forward_only", False, "False for training, True for testing [False]")

# Training configs
flags.DEFINE_integer("max_epoch", 14, "Maximum number of epoch [14]")
flags.DEFINE_integer("num_steps", 14, "Maximum number of epoch [14]")
flags.DEFINE_integer("batch_size", 14, "Maximum number of epoch [14]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_float("max_grad_norm", 10, "Maximum gradient normalization term [10]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_string("dataset", "ptb", "The name of dataset [ptb]")

FLAGS = flags.FLAGS

MODELS = {
  'rcmn': RCMN,
  'rcmn_word': RCMN,
  'rcmn_char': RCMN,
}

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  data_path = "./data/%s" % FLAGS.dataset
  reader = TextReader(data_path)

  with tf.Session() as sess:
    m = MODELS[FLAGS.model]
    model = m(sess, reader, dataset=FLAGS.dataset,
              embed_dim=FLAGS.embed_dim, h_dim=FLAGS.h_dim,
              learning_rate=FLAGS.learning_rate, max_iter=FLAGS.max_iter,
              checkpoint_dir=FLAGS.checkpoint_dir)

    if FLAGS.forward_only:
      model.load(FLAGS.checkpoint_dir)
    else:
      model.train(FLAGS)

    while True:
      text = raw_input(" [*] Enter text to test: ")
      model.sample(5, text)

if __name__ == '__main__':
  tf.app.run()
