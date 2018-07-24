# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:24:49 2018

@author: kaok
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import tensorflow as tf
import numpy as np

import pandas as pd

#dependencies for cloudml
import argparse
import json
import logging
import os

from tensorflow.contrib import layers
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
import util
from util import override_if_not_in_args

# Hyper-parameters
#HIDDEN1 = 1024  # Number of units in hidden layer 1.
#HIDDEN2 = 1024  # Number of units in hidden layer 2.

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
#NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
#IMAGE_SIZE = 28
#IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def create_model():
    """Factory method that creates model to be used by generic task.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.002)
    args, task_args = parser.parse_known_args()

    override_if_not_in_args('--max_steps', '20000000', task_args)
    override_if_not_in_args('--batch_size', '256', task_args)
    override_if_not_in_args('--eval_set_size', '10000', task_args)
    override_if_not_in_args('--eval_interval_secs', '1', task_args)
    override_if_not_in_args('--log_interval_secs', '1', task_args)
    override_if_not_in_args('--min_train_eval_rate', '1', task_args)
    
    model_params = dict(V=68, H=1024, softmax_ns=68, num_layers=2, learning_rate=0.002)

    return Model(**model_params), task_args

class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.examples = None
    self.train = None
    self.global_step = None
    self.metric_updates = []
    self.metric_values = []
    self.keys = None
    self.predictions = []

class GraphIntermediates(object):
  """Holder of tensors passed from one batch to the next."""

  def __init__(self):
    self.loss = None
    self.final_h = []

def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.
    Args:
      X: [m,n,k]
      W: [k,l]
    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.
    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.
    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers
    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cells = []
    for _ in range(num_layers):
      cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
      cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)


# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class Model(object):
    #def __init__(self, learning_rate, hidden1, hidden2):
    #self.learning_rate = learning_rate
    #self.hidden1 = hidden1
    #self.hidden2 = hidden2
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.
        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.
        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V=68, H=1024, softmax_ns=68, num_layers=2, learning_rate=0.002):
        # Model structure; these need to be fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            # Number of samples for sampled softmax.
            self.softmax_ns = softmax_ns

            #self.learning_rate_ = tf.placeholder(tf.float32, [], name="learning_rate")
            self.learning_rate_ = learning_rate

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 1.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def build_graph(self, data_paths, batch_size, max_time, is_training):
        """Construct the core RNNLM graph, needed for any use of the model.
        This should include:
        - Placeholders for input tensors (input_w_, initial_h_, target_y_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)
        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).
        You shouldn't include training or sampling functions here; you'll do
        this in BuildTrainGraph and BuildSampleGraph below.
        We give you some starter definitions for input_w_ and target_y_, as
        well as a few other tensors that might help. We've also added dummy
        values for initial_h_, logits_, and loss_ - you should re-define these
        in your code as the appropriate tensors.
        See the in-line comments for more detail.
        """
        tensors = GraphReferences()
        to_pass = GraphIntermediates()
        
        #_, tensors.examples = util.read_examples(
                #data_paths,
                #batch_size,
                #shuffle=is_training,
                #num_epochs=None if is_training else 2)
        
        # Input ids, with dynamic shape depending on input.
        # Should be shape [batch_size, max_time] and contain integer word indices.
        self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")
        self.initial_h_ = None
        self.final_h_ = None
        # Overwrite this with an actual Tensor of shape
        # [batch_size, max_time, V].
        self.logits_ = None
        #tf.placeholder(tf.int32, [self.batch_size_, self.max_time_, self.V], name="logits")
        #tf.Variable(tf.random_normal([self.batch_size_, self.max_time_, self.V]), name="logits")

        # Should be the same shape as inputs_w_
        self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")
        #tf.placeholder(tf.int32, [None, None], name="y")

        # Replace this with an actual loss function
        self.loss_ = None
        #tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_, logits=self.logits_),0)#don't forget last parameter

        # Get dynamic shape info from inputs
        with tf.name_scope("batch_size"):
            self.batch_size_ = tf.shape(self.input_w_)[0]
        with tf.name_scope("max_time"):
            self.max_time_ = tf.shape(self.input_w_)[1]

        #self.batch_size_ = batch_size
        #self.max_time_ = max_time
        #_, tensors.examples = util.read_examples(
                #data_paths,
                #batch_size=self.batch_size_,
                #shuffle=is_training,
                #num_epochs=None if is_training else 2)
        
        # Get sequence length from input_w_.
        # TL;DR: pass this to dynamic_rnn.
        # This will be a vector with elements ns[i] = len(input_w_[i])
        # You can override this in feed_dict if you want to have different-length
        # sequences in the same batch, although you shouldn't need to for this
        # assignment.
        self.ns_ = tf.tile([self.max_time_], [self.batch_size_, ], name="ns")#update this for project

        
        with tf.name_scope("embedding_layer"):
            self.W_in_ = tf.get_variable("W_in", shape=[self.V, self.H], 
                                         initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
            self.x_ = tf.nn.embedding_lookup(self.W_in_, self.input_w_)



        # Construct RNN/LSTM cell and recurrent layer.
        with tf.name_scope("recurrent_layer"):
            self.cell_ = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
            self.initial_h_ = self.cell_.zero_state(self.batch_size_,tf.float32)
            self.outputs_, self.final_h_ = tf.nn.dynamic_rnn(self.cell_, inputs=self.x_, 
                                                              sequence_length=self.ns_, initial_state=self.initial_h_,
                                                       dtype=tf.float32)

        with tf.name_scope("softmax_output_layer"):
            self.W_out_ = tf.get_variable("W_out", shape=[self.H, self.V], 
                                          initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0))

            self.b_out_ = tf.get_variable("b_out", shape=[self.V,], 
                                          initializer = tf.zeros_initializer())

            self.logits_ = tf.add(matmul3d(self.outputs_, self.W_out_), self.b_out_, name="logits")



        ## Loss computation (true loss, for prediction)
        #with tf.name_scope("loss_computation"):
            #per_example_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_y_, 
                                                                               #logits=self.logits_, 
                                                                               #name="per_example_loss")
            #self.loss_ = tf.reduce_mean(per_example_loss_, name="loss")
        with tf.name_scope("loss_computation"):
            per_example_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_y_, logits=self.logits_, name="per_example_loss")
            self.loss_ = tf.reduce_mean(per_example_loss_, name="loss")

        # Add to the Graph the Ops that calculate and apply gradients.
        # Replace this with an actual training op
        self.train_step_ = None
        # Replace this with an actual loss function
        self.train_loss_ = None

        with tf.name_scope("training_loss_function"):
            per_example_train_loss_ = tf.nn.sampled_softmax_loss(weights=tf.transpose(self.W_out_), biases=self.b_out_, 
                                                                 labels=tf.reshape(self.target_y_, 
                                                                                   [self.batch_size_*self.max_time_,1]),
                                                                 inputs=tf.reshape(self.outputs_, 
                                                                                   [self.batch_size_*self.max_time_,self.H]), 
                                                                 num_sampled=self.softmax_ns, num_classes=self.V, 
                                                                 name="per_example_sampled_softmax_loss")
            #partition_strategy="div" ???
            self.train_loss_ = tf.reduce_mean(per_example_train_loss_, name="sampled_softmax_loss")

        with tf.name_scope("optimizer_and_training_op"):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer_ = tf.train.AdamOptimizer(learning_rate=self.learning_rate_)
            gradients, v = zip(*optimizer_.compute_gradients(self.train_loss_))
            gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm_)
            self.train_step_ = optimizer_.apply_gradients(zip(gradients, v),global_step=global_step)

        self.pred_samples_ = None
        
        with tf.name_scope("sampling_ops"):
            self.pred_samples_ = tf.multinomial(tf.reshape(self.logits_, [-1,self.logits_.get_shape()[-1]]), 
                                                1, name="pred_samples")
            self.pred_samples_ = tf.reshape(self.pred_samples_, [self.batch_size_, self.max_time_, 1])
            
        if is_training:
            tensors.train = self.train_step_
            tensors.global_step = global_step
            loss_value = self.train_loss_
            #use_dropout = True
        else:
            tensors.global_step = tf.Variable(0, name='global_step', trainable=False)
            loss_value = self.loss_
            #use_dropout = False
            #tensors.train = tf.no_op()
        
        # Add means across all batches.
        loss_updates, loss_op = util.loss(loss_value)
        #accuracy_updates, accuracy_op = util.accuracy(self.logits_, parsed['labels'])
        
        if not is_training:
          # Remove this if once Tensorflow 0.12 is standard.
          try:
            #tf.contrib.deprecated.scalar_summary('accuracy', accuracy_op)
            tf.contrib.deprecated.scalar_summary('loss', loss_op)
          except AttributeError:
            #tf.scalar_summary('accuracy', accuracy_op)
            tf.scalar_summary('loss', loss_op)
    
        #tensors.metric_updates = loss_updates + accuracy_updates
        #tensors.metric_values = [loss_op, accuracy_op]
        tensors.metric_updates = loss_updates
        tensors.metric_values = [loss_op]
        to_pass.loss = loss_value
        to_pass.final_h = self.final_h_
        to_pass.inital_h = self.initial_h_
        return tensors, to_pass

    def build_train_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, is_training=True)

    def build_eval_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, is_training=False)

    def export(self, last_checkpoint, output_dir, words_to_ids, ids_to_words):
        """Builds a prediction graph and xports the model.
        Args:
          last_checkpoint: The latest checkpoint from training.
          output_dir: Path to the folder to be used to output the model.
        """
        logging.info('Exporting prediction graph to %s', output_dir)
        with tf.Session(graph=tf.Graph()) as sess:
            try:
                init_op = tf.global_variables_initializer()
            except AttributeError:
                init_op = tf.initialize_all_variables()
            sess.run(init_op)
            trained_saver = tf.train.Saver()
            trained_saver.restore(sess, last_checkpoint)
            
            num_samples = 10
            max_steps = 75
            w = np.repeat([[words_to_ids.get('<SOR>')]], num_samples, axis=0)
            h = sess.run(self.initial_h_, {self.input_w_: w})
            for i in range(max_steps):
                h, y = sample_step(self, sess, w[:,-1:], h)
                w = np.hstack((w,y))
            
            y = []
            for row in w:
                new_review = ""
                for i, word_id in enumerate(row):
                    print(ids_to_words[word_id], end="")
                    new_review = new_review + ids_to_words[word_id]
                    if (i != 0) and (word_id == words_to_ids.get("<EOR>")):
                        break
                print("")
                y.append(new_review)
            #add a way to save these predictions
            
            outputs = {"y": tf.saved_model.utils.build_tensor_info(y)}
            predict_signature_def = signature_def_utils.build_signature_def(
                    outputs=outputs, 
                    signature_constants.PREDICT_METHOD_NAME)
            
            build = builder.SavedModelBuilder(os.path.join(output_dir, 'saved_model'))
            build.add_meta_graph_and_variables(
                    sess, [tag_constants.SERVING],
                    signature_def_map={
                            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                predict_signature_def
                                },
              assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
            build.save()
            
#    def export(self, last_checkpoint, output_dir):
#        """Builds a prediction graph and xports the model.
#        Args:
#          last_checkpoint: The latest checkpoint from training.
#          output_dir: Path to the folder to be used to output the model.
#        """
#        logging.info('Exporting prediction graph to %s', output_dir)
#        with tf.Session(graph=tf.Graph()) as sess:
#          # Build and save prediction meta graph and trained variable values.
#          input_signatures, output_signatures = self.build_prediction_graph()
#          # Remove this if once Tensorflow 0.12 is standard.
#          try:
#            init_op = tf.global_variables_initializer()
#          except AttributeError:
#            init_op = tf.initialize_all_variables()
#          sess.run(init_op)
#          trained_saver = tf.train.Saver()
#          trained_saver.restore(sess, last_checkpoint)
#    
#          predict_signature_def = signature_def_utils.build_signature_def(
#              input_signatures, output_signatures,
#              signature_constants.PREDICT_METHOD_NAME)
#          # Create a saver for writing SavedModel training checkpoints.
#          build = builder.SavedModelBuilder(
#              os.path.join(output_dir, 'saved_model'))
#          build.add_meta_graph_and_variables(
#              sess, [tag_constants.SERVING],
#              signature_def_map={
#                  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                      predict_signature_def
#              },
#              assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
#          build.save()

    def build_prediction_graph(self):
        """Builds prediction graph and registers appropriate endpoints."""
        examples = tf.placeholder(tf.string, shape=(None,))
        features = {
            'image': tf.FixedLenFeature(
                shape=[IMAGE_PIXELS], dtype=tf.float32),
            'key': tf.FixedLenFeature(
                shape=[], dtype=tf.string),
        }
    
        parsed = tf.parse_example(examples, features)
        images = parsed['image']
        keys = parsed['key']
    
        # Build a Graph that computes predictions from the inference model.
        logits = inference(images, self.hidden1, self.hidden2)
        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax, 1)
        
        
        
        #max_steps = 100
        #num_samples = 10
        #random_seed = 42
        
        #model_params = dict(V=68, H=1024, softmax_ns=68, num_layers=2)
        
        #lm = Model(**model_params)
        #lm.
        #w = np.repeat([[words_to_ids.get('<SOR>')]], num_samples, axis=0)
    
        # Mark the inputs and the outputs
        # Marking the input tensor with an alias with suffix _bytes. This is to
        # indicate that this tensor value is raw bytes and will be base64 encoded
        # over HTTP.
        # Note that any output tensor marked with an alias with suffix _bytes, shall
        # be base64 encoded in the HTTP response. To get the binary value, it
        # should be base64 decoded.
        input_signatures = {}
        predict_input_tensor = meta_graph_pb2.TensorInfo()
        predict_input_tensor.name = examples.name
        predict_input_tensor.dtype = examples.dtype.as_datatype_enum
        input_signatures['example_bytes'] = predict_input_tensor
    
        tf.add_to_collection('inputs',
                             json.dumps({
                                 'examples_bytes': examples.name
                             }))
        tf.add_to_collection('outputs',
                             json.dumps({
                                 'key': keys.name,
                                 'prediction': prediction.name,
                                 'scores': softmax.name
                             }))
        output_signatures = {}
        outputs_dict = {'key': keys.name,
                        'prediction': prediction.name,
                        'scores': softmax.name}
        for key, val in outputs_dict.iteritems():
          predict_output_tensor = meta_graph_pb2.TensorInfo()
          predict_output_tensor.name = val
          for placeholder in [keys, prediction, softmax]:
            if placeholder.name == val:
              predict_output_tensor.dtype = placeholder.dtype.as_datatype_enum
          output_signatures[key] = predict_output_tensor
        return input_signatures, output_signatures

    #def format_metric_values(self, metric_values):
        #"""Formats metric values - used for logging purpose."""
        #return 'loss: %.3f, accuracy: %.3f' % (metric_values[0], metric_values[1])

    def format_metric_values(self, metric_values):
        """Formats metric values - used for logging purpose."""
        return 'loss: %.3f' % (metric_values[0])

    def format_prediction_values(self, prediction):
        """Formats prediction values - used for writing batch predictions as csv."""
        return '%.3f' % (prediction[0])



        #### END(YOUR CODE) ####
def parse_examples(examples):
  feature_map = {
      'labels':
          tf.FixedLenFeature(
              shape=[], dtype=tf.int64, default_value=[-1]),
      'images':
          tf.FixedLenFeature(
              shape=[IMAGE_PIXELS], dtype=tf.float32),
  }
  return tf.parse_example(examples, features=feature_map)

def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  hidden1 = layers.fully_connected(images, hidden1_units)
  hidden2 = layers.fully_connected(hidden1, hidden2_units)
  return layers.fully_connected(hidden2, NUM_CLASSES, activation_fn=None)

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')

#def training(loss_op, learning_rate):
#  """Sets up the training Ops.
#  Creates a summarizer to track the loss over time in TensorBoard.
#  Creates an optimizer and applies the gradients to all trainable variables.
#  The Op returned by this function is what must be passed to the
#  `sess.run()` call to cause the model to train.
#  Args:
#    loss_op: Loss tensor, from loss().
#    learning_rate: The learning rate to use for gradient descent.
#  Returns:
#    A pair consisting of the Op for training and the global step.
#  """
#  # Create the gradient descent optimizer with the given learning rate.
#  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#  # Create a variable to track the global step.
#  global_step = tf.Variable(0, name='global_step', trainable=False)
#  # Use the optimizer to apply the gradients that minimize the loss
#  # (and also increment the global step counter) as a single training step.
#  train_op = optimizer.minimize(loss_op, global_step=global_step)
#  return train_op, global_step
###
def sample_step(lm, session, input_w, initial_h):
    """Run a single RNN step and return sampled predictions.
  
    Args:
      lm : rnnlm.RNNLM
      session: tf.Session
      input_w : [batch_size] vector of indices
      initial_h : [batch_size, hidden_dims] initial state
    
    Returns:
      final_h : final hidden state, compatible with initial_h
      samples : [batch_size, 1] vector of indices
    """
    # Reshape input to column vector
    input_w = np.array(input_w, dtype=np.int32).reshape([-1,1])

    # Run sample ops
    feed_dict = {lm.input_w_:input_w, lm.initial_h_:initial_h}
    final_h, samples = session.run([lm.final_h_, lm.pred_samples_], feed_dict=feed_dict)

    # Note indexing here: 
    #   [batch_size, max_time, 1] -> [batch_size, 1]
    return final_h, samples[:,-1,:]

def generate_text(trained_filename, model_params, words_to_ids, ids_to_words):
    # Same as above, but as a batch
    #max_steps = 20
    max_steps = 50
    num_samples = 10
    random_seed = 42
    
    lm = Model(**model_params)
    lm.BuildCoreGraph()
    lm.BuildSamplerGraph()
    
    with lm.graph.as_default():
        saver = tf.train.Saver()
    
    with tf.Session(graph=lm.graph) as session:
        # Seed RNG for repeatability
        tf.set_random_seed(random_seed)
        
        # Load the trained model
        saver.restore(session, trained_filename)
    
        # Make initial state for a batch with batch_size = num_samples
        #w = np.repeat([[vocab.START_ID]], num_samples, axis=0)
        w = np.repeat([[words_to_ids.get('<SOR>')]], num_samples, axis=0)
        h = session.run(lm.initial_h_, {lm.input_w_: w})
        # take one step for each sequence on each iteration 
        for i in range(max_steps):
            h, y = sample_step(lm, session, w[:,-1:], h)
            w = np.hstack((w,y))
    
        # Print generated sentences
        for row in w:
            for i, word_id in enumerate(row):
                #print(vocab.id_to_word[word_id], end=" ")
                print(ids_to_words[word_id], end="")
                #if (i != 0) and (word_id == vocab.START_ID):
                if (i != 0) and (word_id == words_to_ids.get("<EOR>")):
                    break
            print("")
#self.tensors = self.model.build_eval_graph(self.eval_data_paths, self.eval_batch_size)