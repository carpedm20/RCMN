import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

def conv2d(inputs, out_dim, k_w, k_h, stddev=0.02, name="conv2d", bias_start=0):
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]

  with tf.variable_scope(name):
    input_ = tf.concat(3, inputs)

    w = tf.get_variable('conv_k', [k_w, k_h, input_.get_shape()[-1], out_dim],
                        initializer=tf.contrib.layers.xavier_initializer_conv2d())

    if len(inputs) == 1:
      conv = tf.nn.conv2d(inputs[0], w, [1, 1, 1, 1], "SAME")
    else:
      conv = tf.nn.conv2d(input_, w, [1, 1, 1, 1], "SAME")

    if bias_start == None:
      return conv

    bias = tf.get_variable("conv_b", [out_dim],
                           initializer=tf.constant_initializer(0.0))
    return conv + bias + bias_start

def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Code modification of tensorflw/python/ops/rnn_cell.py
  Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size],
                             initializer=tf.contrib.layers.xavier_initializer())
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
  return res + bias_term
