import tensorflow as tf
from tensorflow.models.rnn import rnn_cell


def conv2d(inputs, out_dim, k_w, k_h, stddev=0.02, name="conv2d", bias_start=0):
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]

  with tf.variable_scope(name):
    input_ = tf.concat(3, inputs)

    w = tf.get_variable('conv_k', [k_w, k_h, input_.get_shape()[-1], out_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))

    if len(inputs) == 1:
      conv = tf.nn.conv2d(inputs[0], w, [1, 1, 1, 1], "SAME")
    else:
      conv = tf.nn.conv2d(input_, w, [1, 1, 1, 1], "SAME")

    if bias_start == None:
      return conv

    bias = tf.get_variable("conv_b", [out_dim],
                           initializer=tf.constant_initializer(0.0))
    return conv + bias + bias_start
