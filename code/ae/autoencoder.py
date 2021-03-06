from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin

import numpy as np
import tensorflow as tf
from utils.data import fill_feed_dict_ae, read_data_sets_pretraining
from utils.data import read_data_sets, fill_feed_dict
from utils.flags import FLAGS
from utils.eval import loss_supervised, evaluation, do_eval
from utils.utils import tile_raster_images


class AutoEncoder(object):
  _weights_str = "weights{0}"
  _biases_str = "biases{0}"

  def __init__(self, shape, sess):
    self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
    self.__num_hidden_layers = len(self.__shape) - 2

    self.__variables = {}
    self.__sess = sess

    self._setup_variables()

  @property
  def shape(self):
    return self.__shape

  @property
  def num_hidden_layers(self):
    return self.__num_hidden_layers

  @property
  def session(self):
    return self.__sess

  def __getitem__(self, item):
    return self.__variables[item]

  def __setitem__(self, key, value):
    self.__variables[key] = value

  def _setup_variables(self):
    with tf.name_scope("autoencoder_variables"):
      for i in xrange(self.__num_hidden_layers + 1):
        # Train weights
        name_w = self._weights_str.format(i + 1)
        w_shape = (self.__shape[i], self.__shape[i + 1])
        a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                   name=name_w,
                                   trainable=True)
        # Train biases
        name_b = self._biases_str.format(i + 1)
        b_shape = (self.__shape[i + 1],)
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

        if i < self.__num_hidden_layers:
          # Hidden layer fixed weights (after pretraining before fine tuning)
          self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                                                name=name_w + "_fixed",
                                                trainable=False)

          # Hidden layer fixed biases
          self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                                                name=name_b + "_fixed",
                                                trainable=False)

          # Pretraining output training biases
          name_b_out = self._biases_str.format(i + 1) + "_out"
          b_shape = (self.__shape[i],)
          b_init = tf.zeros(b_shape)
          self[name_b_out] = tf.Variable(b_init,
                                         trainable=True,
                                         name=name_b_out)

  def _w(self, n, suffix=""):
    return self[self._weights_str.format(n) + suffix]

  def _b(self, n, suffix=""):
    return self[self._biases_str.format(n) + suffix]

  def get_variables_to_init(self, n):
    assert n > 0
    assert n <= self.__num_hidden_layers + 1

    vars_to_init = [self._w(n), self._b(n)]

    if n <= self.__num_hidden_layers:
      vars_to_init.append(self._b(n, "_out"))

    if 1 < n <= self.__num_hidden_layers:
      vars_to_init.append(self._w(n - 1, "_fixed"))
      vars_to_init.append(self._b(n - 1, "_fixed"))

    return vars_to_init

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
    return y

  def pretrain_net(self, input_pl, n, is_target=False):
    assert n > 0
    assert n <= self.__num_hidden_layers

    last_output = input_pl
    for i in xrange(n - 1):
      w = self._w(i + 1, "_fixed")
      b = self._b(i + 1, "_fixed")

      last_output = self._activate(last_output, w, b)

    if is_target:
      return last_output

    last_output = self._activate(last_output, self._w(n), self._b(n))

    out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                         transpose_w=True)
    out = tf.maximum(out, 1.e-9)
    out = tf.minimum(out, 1 - 1.e-9)
    return out

  def supervised_net(self, input_pl):
    last_output = input_pl

    for i in xrange(self.__num_hidden_layers + 1):
      # Fine tuning will be done on these variables
      w = self._w(i + 1)
      b = self._b(i + 1)

      last_output = self._activate(last_output, w, b)

    return last_output

def training(loss, learning_rate, loss_key=None):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.minimize(loss)
  return train_op


def loss_x_entropy(output, target):
  with tf.name_scope("xentropy_loss"):
      net_output_tf = tf.convert_to_tensor(output, name='input')
      target_tf = tf.convert_to_tensor(target, name='target')
      cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'),
                                    target_tf),
                             tf.multiply(tf.log(1 - net_output_tf),
                                    (1 - target_tf)))
      return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                 name='xentropy_mean')


def main_unsupervised():
  with tf.Graph().as_default() as g:
    sess = tf.Session()
    num_hidden = FLAGS.num_hidden_layers
    ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                        for j in xrange(num_hidden)]
    ae_shape = [FLAGS.image_pixels] + ae_hidden_shapes + [FLAGS.num_classes]
    ae = AutoEncoder(ae_shape, sess)
    data = read_data_sets_pretraining(FLAGS.data_dir)
    num_train = data.train.num_examples
    learning_rates = {j: getattr(FLAGS, "pre_layer{0}_learning_rate".format(j + 1)) for j in xrange(num_hidden)}
    noise = {j: getattr(FLAGS, "noise_{0}".format(j + 1)) for j in xrange(num_hidden)}

    for i in xrange(len(ae_shape) - 2):
      n = i + 1
      with tf.variable_scope("pretrain"):
        input_ = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, ae_shape[0]), name='ae_input_pl')
        target_ = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, ae_shape[0]), name='ae_target_pl')
        layer = ae.pretrain_net(input_, n)
        with tf.name_scope("target"):
          target_for_loss = ae.pretrain_net(target_, n, is_target=True)

        loss = loss_x_entropy(layer, target_for_loss)
        train_op = training(loss, learning_rates[i], i)
        vars_to_init = ae.get_variables_to_init(n)
        sess.run(tf.initialize_variables(vars_to_init))

        print("\n\n")
        print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
        print("|---------------|---------------|---------|----------|")

        for step in xrange(FLAGS.pretraining_epochs):
          for istep in xrange(int(num_train / FLAGS.batch_size)):
            feed_dict = fill_feed_dict_ae(data.train, input_, target_, noise[i])
            loss_summary, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            if istep % 100 == 0:
              output = "| {0:>13} | {1:13.4f} | Layer {2} | Epoch {3}  |".format(istep, loss_value, n, step + 1)
              print(output)
  return ae


def main_supervised(ae):
  with ae.session.graph.as_default():
    sess = ae.session
    input_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
                                                 FLAGS.image_pixels),
                              name='input_pl')
    logits = ae.supervised_net(input_pl)

    data = read_data_sets(FLAGS.data_dir)
    num_train = data.train.num_examples
    labels_placeholder = tf.placeholder(tf.int32,
                                        shape=FLAGS.batch_size,
                                        name='target_pl')

    loss = loss_supervised(logits, labels_placeholder)
    train_op = training(loss, FLAGS.supervised_learning_rate)
    eval_correct = evaluation(logits, labels_placeholder)
    vars_to_init = ae.get_variables_to_init(ae.num_hidden_layers + 1)
    sess.run(tf.initialize_variables(vars_to_init))

    steps = FLAGS.finetuning_epochs * int(num_train/FLAGS.batch_size)
    for step in xrange(steps):
      start_time = time.time()
      feed_dict = fill_feed_dict(data.train, input_pl, labels_placeholder)
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      duration = time.time() - start_time
      if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
    do_eval(sess, eval_correct, input_pl, labels_placeholder, data.test)

if __name__ == '__main__':
  ae = main_unsupervised()
  main_supervised(ae)
