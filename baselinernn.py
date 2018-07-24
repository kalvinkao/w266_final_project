# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 12:30:13 2018

@author: kalvi
"""
## Load Dependencies
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

#import json, os, re, shutil, sys, time
import os, shutil, time
from importlib import reload
#import collections, itertools
import unittest
#from IPython.display import display, HTML

# NLTK for NLP utils and corpora
#import nltk

# NumPy and TensorFlow
import numpy as np
import tensorflow as tf
assert(tf.__version__.startswith("1."))

# Helper libraries
from w266_common import utils#, vocabulary, tf_embed_viz

# rnnlm code
import rnnlm; reload(rnnlm)
import rnnlm_test; reload(rnnlm_test)

# packages for extracting data
import pandas as pd
#from glob import glob
#import csv

def make_tensorboard(tf_graphdir="/tmp/artificial_hotel_reviews/a4_graph", V=100, H=1024, num_layers=2):
    reload(rnnlm)
    TF_GRAPHDIR = tf_graphdir
    # Clear old log directory.
    shutil.rmtree(TF_GRAPHDIR, ignore_errors=True)
    
    lm = rnnlm.RNNLM(V=V, H=H, num_layers=num_layers)
    lm.BuildCoreGraph()
    lm.BuildTrainGraph()
    lm.BuildSamplerGraph()
    summary_writer = tf.summary.FileWriter(TF_GRAPHDIR, lm.graph)
    return summary_writer

# Unit Tests
def test_graph():
    reload(rnnlm); reload(rnnlm_test)
    utils.run_tests(rnnlm_test, ["TestRNNLMCore", "TestRNNLMTrain", "TestRNNLMSampler"])

def test_training():
    reload(rnnlm); reload(rnnlm_test)
    th = rnnlm_test.RunEpochTester("test_toy_model")
    th.setUp(); th.injectCode(run_epoch, score_dataset)
    unittest.TextTestRunner(verbosity=2).run(th)

## Training Functions
def run_epoch(lm, session, batch_iterator,
              train=False, verbose=False,
              tick_s=10, learning_rate=None):
    assert(learning_rate is not None)
    start_time = time.time()
    tick_time = start_time  # for showing status
    total_cost = 0.0  # total cost, summed over all words
    total_batches = 0
    total_words = 0

    if train:
        train_op = lm.train_step_
        use_dropout = True
        loss = lm.train_loss_
    else:
        train_op = tf.no_op()
        use_dropout = False  # no dropout at test time
        loss = lm.loss_  # true loss, if train_loss is an approximation

    for i, (w, y) in enumerate(batch_iterator):
        cost = 0.0
        # At first batch in epoch, get a clean intitial state.
        if i == 0:
            h = session.run(lm.initial_h_, {lm.input_w_: w})

        #### YOUR CODE HERE ####
        feed_dict = {lm.input_w_:w,
                     lm.target_y_:y,
                     lm.learning_rate_: learning_rate,
                     lm.use_dropout_: use_dropout,
                     lm.initial_h_:h}
        cost, h, _ = session.run([loss, lm.final_h_, train_op],feed_dict=feed_dict)

        #### END(YOUR CODE) ####
        total_cost += cost
        total_batches = i + 1
        total_words += w.size  # w.size = batch_size * max_time

        ##
        # Print average loss-so-far for epoch
        # If using train_loss_, this may be an underestimate.
        if verbose and (time.time() - tick_time >= tick_s):
            avg_cost = total_cost / total_batches
            avg_wps = total_words / (time.time() - start_time)
            print("[batch {:d}]: seen {:d} words at {:.1f} wps, loss = {:.3f}".format(
                i, total_words, avg_wps, avg_cost))
            tick_time = time.time()  # reset time ticker

    return total_cost / total_batches

def score_dataset(lm, session, ids, name="Data"):
    # For scoring, we can use larger batches to speed things up.
    bi = utils.rnnlm_batch_generator(ids, batch_size=100, max_time=100)
    cost = run_epoch(lm, session, bi, 
                     learning_rate=0.0, train=False, 
                     verbose=False, tick_s=3600)
    print("{:s}: avg. loss: {:.03f}  (perplexity: {:.02f})".format(name, cost, np.exp(cost)))
    return cost

#build a list of list of characters from the 5-star reviews
def preprocess_review_series(review_series):
    review_list = []
    for new_review in review_series:
        clipped_review = new_review[2:-1]
        char_list = list(clipped_review.lower())
        semifinal_review = []
        last_char = ''
        for ascii_char in char_list:
            if ascii_char == '\\' or last_char == '\\':
                pass
            else:
                semifinal_review.append(ascii_char)
            last_char = ascii_char
        final_review = ['<SOR>'] + semifinal_review + ['<EOR>']
        #print(final_review)
        review_list.append(final_review)
    return review_list

def get_review_series(review_path = '/home/kalvin_kao/yelp_challenge_dataset/review.csv'):
    #review_path = '/home/kalvin_kao/yelp_challenge_dataset/review.csv'
    review_df = pd.read_csv(review_path)
    five_star_review_df = review_df[review_df['stars']==5]
    #five_star_review_series = five_star_review_df['text']
    return five_star_review_df['text']

def get_business_list(business_path = '/home/kalvin_kao/yelp_challenge_dataset/business.csv'):
    #business_path = '/home/kalvin_kao/yelp_challenge_dataset/business.csv'
    return pd.read_csv(business_path)

def make_train_test_data(five_star_review_series, training_samples=20000, test_samples=1000):
    #add randomization
    review_list = preprocess_review_series(five_star_review_series)
    training_review_list = [item for sublist in review_list[:training_samples] for item in sublist]
    print(len(training_review_list))
    
    test_review_list = [item for sublist in review_list[training_samples:training_samples+test_samples] for item in sublist]
    return training_review_list, test_review_list

#def make_test_data()
#test_review_list = [item for sublist in review_list[50000:51000] for item in sublist]
    
def make_vocabulary(training_review_list, test_review_list):
    unique_characters = list(set(training_review_list + test_review_list))
    #vocabulary
    char_dict = {w:i for i, w in enumerate(unique_characters)}
    ids_to_words = {v: k for k, v in char_dict.items()}
    return char_dict, ids_to_words

def convert_to_ids(char_dict, review_list):
    #convert to flat (1D) np.array(int) of ids
    review_ids = [char_dict.get(token) for token in review_list]
    return np.array(review_ids)
#training_review_ids = [char_dict.get(token) for token in training_review_list]
#test_review_ids = [char_dict.get(token) for token in test_review_list]
#train_ids = np.array(training_review_ids)
#test_ids = np.array(test_review_ids)

#def run_training(train_ids, test_ids, max_time=100, batch_size=256, learning_rate=0.002, num_epochs=20, V=68, H=1024, softmax_ns=68, num_layers=2, tf_savedir = "/tmp/artificial_hotel_reviews/a4_model"):
def run_training(train_ids, test_ids, max_time=100, batch_size=256, learning_rate=0.002, num_epochs=20, model_params, tf_savedir = "/tmp/artificial_hotel_reviews/a4_model"):
    #V = len(words_to_ids.keys())
    # Training parameters
    ## add parameter sets for each attack/defense configuration
    #max_time = 25
    #batch_size = 100
    #learning_rate = 0.01
    #num_epochs = 10
    
    # Model parameters
    #model_params = dict(V=vocab.size, 
                        #H=200, 
                        #softmax_ns=200,
                        #num_layers=2)
    #model_params = dict(V=len(words_to_ids.keys()), 
                        #H=1024, 
                        #softmax_ns=len(words_to_ids.keys()),
                        #num_layers=2)
    #model_params = dict(V=V, H=H, softmax_ns=softmax_ns, num_layers=num_layers)
    
    #TF_SAVEDIR = "/tmp/artificial_hotel_reviews/a4_model"
    TF_SAVEDIR = tf_savedir
    checkpoint_filename = os.path.join(TF_SAVEDIR, "rnnlm")
    trained_filename = os.path.join(TF_SAVEDIR, "rnnlm_trained")
    
    # Will print status every this many seconds
    #print_interval = 5
    print_interval = 30
    
    lm = rnnlm.RNNLM(**model_params)
    lm.BuildCoreGraph()
    lm.BuildTrainGraph()
    
    # Explicitly add global initializer and variable saver to LM graph
    with lm.graph.as_default():
        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
    # Clear old log directory
    shutil.rmtree(TF_SAVEDIR, ignore_errors=True)
    if not os.path.isdir(TF_SAVEDIR):
        os.makedirs(TF_SAVEDIR)
    
    with tf.Session(graph=lm.graph) as session:
        # Seed RNG for repeatability
        tf.set_random_seed(42)
    
        session.run(initializer)
        
        #check trainable variables
        #variables_names = [v.name for v in tf.trainable_variables()]
        #values = session.run(variables_names)
        #for k, v in zip(variables_names, values):
            #print("Variable: ", k)
            #print("Shape: ", v.shape)
            #print(v)
    
        for epoch in range(1,num_epochs+1):
            t0_epoch = time.time()
            bi = utils.rnnlm_batch_generator(train_ids, batch_size, max_time)
            print("[epoch {:d}] Starting epoch {:d}".format(epoch, epoch))
            # Run a training epoch.
            run_epoch(lm, session, batch_iterator=bi, train=True, verbose=True, tick_s=10, learning_rate=learning_rate)
    
            print("[epoch {:d}] Completed in {:s}".format(epoch, utils.pretty_timedelta(since=t0_epoch)))
        
            # Save a checkpoint
            saver.save(session, checkpoint_filename, global_step=epoch)
        
            ##
            # score_dataset will run a forward pass over the entire dataset
            # and report perplexity scores. This can be slow (around 1/2 to 
            # 1/4 as long as a full epoch), so you may want to comment it out
            # to speed up training on a slow machine. Be sure to run it at the 
            # end to evaluate your score.
            #print("[epoch {:d}]".format(epoch), end=" ")
            #score_dataset(lm, session, train_ids, name="Train set")
            print("[epoch {:d}]".format(epoch), end=" ")
            score_dataset(lm, session, test_ids, name="Test set")
            print("")
        
        # Save final model
        saver.save(session, trained_filename)
        return trained_filename

## Sampling
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

def generate_text(trained_filename, model_params):
    # Same as above, but as a batch
    #max_steps = 20
    max_steps = 50
    num_samples = 10
    random_seed = 42
    
    lm = rnnlm.RNNLM(**model_params)
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
        [char_dict.get(token) for token in test_review_list]
        w = np.repeat([[char_dict.get('<SOR>')]], num_samples, axis=0)
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
                if (i != 0) and (word_id == char_dict.get("<EOR>")):
                    break
            print("")
            
def train_attack_model(training_samples=20000, test_samples=1000, review_path = '/home/kalvin_kao/yelp_challenge_dataset/review.csv'):
    #training_samples=20000
    #test_samples=1000
    #review_path = '/home/kalvin_kao/yelp_challenge_dataset/review.csv'
    five_star_reviews = get_review_series(review_path)
    train_review_list, test_review_list = make_train_test_data(five_star_reviews, training_samples, test_samples)
    words_to_ids, ids_to_words = make_vocabulary(train_review_list, test_review_list)
    train_ids = convert_to_ids(words_to_ids, train_review_list)
    test_ids = convert_to_ids(words_to_ids, test_review_list)
    model_params = dict(V=len(words_to_ids.keys()), 
                            H=1024, 
                            softmax_ns=len(words_to_ids.keys()),
                            num_layers=2)
    trained_filename = run_training(train_ids, test_ids, max_time=100, batch_size=256, learning_rate=0.002, num_epochs=20, model_params, tf_savedir = "/tmp/artificial_hotel_reviews/a4_model")
    return trained_filename, model_params

trained_filename, model_params = train_attack_model(training_samples=20000, test_samples=1000, review_path = 'gs://w266_final_project_kk/data/review.csv')
generate_text(trained_filename, model_params)