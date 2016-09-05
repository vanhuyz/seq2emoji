import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab
from six.moves import range
from sklearn.manifold import TSNE
import MeCab
import pickle

from sequence import Sequence
from load_emojis import load_emojis

mecab = MeCab.Tagger("-Owakati")


# Load dictionary
with open('dictionary/dictionary.pickle', 'rb') as handle:
  dictionary = pickle.load(handle)

with open('dictionary/reverse_dictionary.pickle', 'rb') as handle:
  reverse_dictionary = pickle.load(handle)

with open('dictionary/emoji_dictionary.pickle', 'rb') as handle:
  emoji_dictionary = pickle.load(handle)

with open('dictionary/reverse_emoji_dictionary.pickle', 'rb') as handle:
  reverse_emoji_dictionary = pickle.load(handle)

lstm_size = 64
max_length = 50
embedding_size = 128
vocabulary_size = len(dictionary)
emoji_size = len(emoji_dictionary)

def seq2id(sequence):
  words = mecab.parse(sequence).split()
  input_words = words[:-1]
  label_id = emoji_dictionary.get(sequence[-1], -1)
  input_ids = list(map(lambda word: dictionary.get(word,0), input_words))
  input_length = len(input_ids)
  if input_length > max_length:
    input_length = max_length
    input_ids = input_ids[-max_length:]
  else:
    input_ids = input_ids + [0]*(max_length - input_length)
  
  return input_ids, label_id, input_length

def predict_emoji(predictions):
  return reverse_emoji_dictionary[np.argmax(predictions[0])]

def is_likely(predictions):
  return np.max(predictions[0]) > 0.1

def seq2emoji(test_sequence):
  graph = tf.Graph()

  # Graph
  with graph.as_default():

    inputs = tf.placeholder(tf.int32, shape=[None, max_length])
    labels = tf.placeholder(tf.float32, shape=[None, emoji_size])
    sequence_lengths = tf.placeholder(tf.int32, shape=(None,))

    print(inputs.get_shape())
    
    embedding = tf.get_variable("embedding", shape=[vocabulary_size, embedding_size], dtype=tf.float32)
    
    embed_inputs = tf.nn.embedding_lookup(embedding, inputs)
    print(embed_inputs.get_shape())
    
    cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
    with tf.variable_scope("train_valid"):
      outputs, state = tf.nn.dynamic_rnn(cell,
                        embed_inputs,
                        dtype=tf.float32,
                        sequence_length=sequence_lengths
                        )

      
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([lstm_size, emoji_size], -0.1, 0.1), name='w')
    b = tf.Variable(tf.zeros([emoji_size]), name='b')

    logits = tf.nn.xw_plus_b(state.h, w, b)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    train_predictions = tf.nn.softmax(logits)
      
    # Validation eval  
    valid_inputs = tf.placeholder(tf.int32, shape=[None, max_length])
    valid_sequence_lengths = tf.placeholder(tf.int32, shape=(None,))
    valid_embed_inputs = tf.nn.embedding_lookup(embedding, valid_inputs)
    with tf.variable_scope("train_valid", reuse = True):
      valid_outputs, valid_state = tf.nn.dynamic_rnn(cell,
                                                     valid_embed_inputs,
                                                     dtype=tf.float32,
                                                     sequence_length=valid_sequence_lengths
                                                     )
    valid_predictions = tf.nn.softmax(tf.nn.xw_plus_b(valid_state.h, w, b))

    # Test
    test_inputs = tf.placeholder(tf.int32, shape=[None, max_length])
    test_sequence_lengths = tf.placeholder(tf.int32, shape=(None,))
    test_embed_inputs = tf.nn.embedding_lookup(embedding, test_inputs)
    with tf.variable_scope("train_valid", reuse = True):
      test_outputs, test_state = tf.nn.dynamic_rnn(cell,
                                                     test_embed_inputs,
                                                     dtype=tf.float32,
                                                     sequence_length=test_sequence_lengths
                                                     )
    test_predictions = tf.nn.softmax(tf.nn.xw_plus_b(test_state.h, w, b))
    saver = tf.train.Saver()

  # Test
  with tf.Session(graph=graph) as sess:
    # Restore variables from disk.
    saver.restore(sess, "model/model.ckpt")
    print("Model restored.")
      
    #test_sequence = 'これは使えるかな？ '
    test_ids, test_label_id, test_sequence_length = seq2id(test_sequence)
    
    test_feed_dict = { test_inputs: [test_ids], test_sequence_lengths: [test_sequence_length] }
    _test_predictions = test_predictions.eval(feed_dict=test_feed_dict)
    print('Test   :')
    print('  Input      : ', test_sequence[:-1])
    print('  Prediction : ', predict_emoji(_test_predictions))
    print('  Likelihood : ', np.max(_test_predictions[0]))
  return _test_predictions


# API
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api')
def main():
  sequence = request.args.get('seq')
  predictions = seq2emoji("%s　" % sequence)
  emoji = predict_emoji(predictions)
  likely = is_likely(predictions)
  return jsonify(emoji=emoji, likely=str(likely))

if __name__ == "__main__":
  app.run(host= '0.0.0.0')

