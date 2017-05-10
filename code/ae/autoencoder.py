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

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads,0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def main_unsupervised():
  with tf.Graph().as_default() as g:
    num_hidden = FLAGS.num_hidden_layers
    ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                        for j in xrange(num_hidden)]
    ae_shape = [FLAGS.image_pixels] + ae_hidden_shapes + [FLAGS.num_classes]
    data = read_data_sets_pretraining(FLAGS.data_dir)
    num_train = data.train.num_examples
    learning_rates = {j: getattr(FLAGS, "pre_layer{0}_learning_rate".format(j + 1)) for j in xrange(num_hidden)}
    noise = {j: getattr(FLAGS, "noise_{0}".format(j + 1)) for j in xrange(num_hidden)}

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False))
    ae = AutoEncoder(ae_shape, sess)
    train_steps = int(num_train / (FLAGS.num_GPUs*FLAGS.batch_size))

    for i in xrange(num_hidden):
      n = i + 1
      optimizer = tf.train.GradientDescentOptimizer(learning_rates[i])
      vars_to_init = ae.get_variables_to_init(n)
      sess.run(tf.initialize_variables(vars_to_init))
      tower_grads = []
      #losses = []
      with tf.variable_scope("pretrain"):
        input_ = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size*FLAGS.num_GPUs, ae_shape[0]), name='ae_input_pl')
        target_ = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size*FLAGS.num_GPUs, ae_shape[0]), name='ae_target_pl')
        for j in xrange(FLAGS.num_GPUs):
          with tf.device('/gpu:0'):         
            layer = ae.pretrain_net(input_[FLAGS.batch_size*j:FLAGS.batch_size*(j+1)], n)
            with tf.name_scope("target"):
              target_for_loss = ae.pretrain_net(target_[FLAGS.batch_size*j:FLAGS.batch_size*(j+1)], n, is_target=True)
            loss = loss_x_entropy(layer, target_for_loss)
            tf.get_variable_scope().reuse_variables()
            #losses.append(loss)
            localgrads = optimizer.compute_gradients(loss, var_list=[ae._w(n), ae._b(n)])
            tower_grads.append(localgrads)

      print("\n\n")
      print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
      print("|---------------|---------------|---------|----------|")

      tf.train.start_queue_runners(sess=sess)

      for step in xrange(FLAGS.pretraining_epochs):
        for istep in xrange(train_steps):
          feed_dict = fill_feed_dict_ae(data.train, input_, target_, noise[i])
          grads = average_gradients(tower_grads)
          train_op = optimizer.apply_gradients(grads)
          #total_loss = tf.add_n(losses, name='total_loss')
          loss_summary, loss_value = sess.run([train_op,loss], feed_dict = feed_dict)
          #loss_summary = sess.run(loss, feed_dict = feed_dict)
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
