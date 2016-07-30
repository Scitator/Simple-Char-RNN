'''
Created on 30 July 2016
@author: Kolesnikov Sergey
'''
# raw version
import tensorflow as tf
import numpy as np
from data_io import *


# hyperparameters
learning_rate   = 1e-1
sequence_length = 25 # number of step_s to unroll the RNN for
hidden_size     = 100 # size of hidden layer of neurons
max_steps = 1000
print_step = 1000

x = tf.placeholder(tf.float32, [None, vocabulary_size])
y = tf.placeholder(tf.float32, [None, vocabulary_size])
h = tf.placeholder(tf.float32, [1, hidden_size])

# model parameters
with tf.variable_scope("rnn", reuse=False) as rnn_scope:
    # weights
    # input to hidden
    W_xh = tf.get_variable("W_xh", [vocabulary_size, hidden_size],
                           tf.float32)
    # hidden to hidden
    W_hh = tf.get_variable("W_hh", [hidden_size, hidden_size],
                           tf.float32)
    # hidden to output
    W_hy = tf.get_variable("W_hy", [hidden_size, vocabulary_size],
                           tf.float32)

    # biases
    # hidden bias
    b_h = tf.get_variable("b_h", [hidden_size],
                          tf.float32)
    # output bias
    b_y = tf.get_variable("b_y", [vocabulary_size],
                          tf.float32)

    init = tf.initialize_all_variables()

# rnn model
with tf.variable_scope("rnn", reuse=True) as rnn_scope:
    # weights
    # input to hidden
    W_xh = tf.get_variable("W_xh", [vocabulary_size, hidden_size],
                           tf.float32, initializer=init)
    # hidden to hidden
    W_hh = tf.get_variable("W_hh", [hidden_size, hidden_size],
                           tf.float32, initializer=init)
    # hidden to output
    W_hy = tf.get_variable("W_hy", [hidden_size, vocabulary_size],
                           tf.float32, initializer=init)

    # biases
    # hidden bias
    b_h = tf.get_variable("b_h", [hidden_size],
                          tf.float32, initializer=init)
    # output bias
    b_y = tf.get_variable("b_y", [vocabulary_size],
                          tf.float32, initializer=init)


    # crar-rnn model (aka forward pass)
    # as we don't need to do backward pass it is earier
    y_s = []
    h_prev = h
    for x_s in tf.split(0, sequence_length, x):
        h_prev = tf.tanh(tf.matmul(x_s, W_xh) +
                         tf.matmul(h_prev, W_hh) + b_h)
        y_ = tf.matmul(h_prev, W_hy) + b_y
        y_s.append(y_)

    # model optimizers (aka backward pass)
    y_pred = tf.concat(0, y_s)
    p_s = tf.nn.softmax(y_s[-1])
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

def np_to_tf_one_hot(x):
    return np.eye(vocabulary_size)[x]

with tf.Session() as sess:
    sess.run(init)
    n, p = 0, 0
    while n <= max_steps:
        if p + sequence_length + 1 >= len(data) or n == 0:
            h_prev_ = np.zeros([1, hidden_size])
            p = 0

        x_ = [char_to_index[ch] for ch in data[p:p+sequence_length]]
        y_ = [char_to_index[ch] for ch in data[p+1:p+sequence_length+1]]

        x_ = np_to_tf_one_hot(x_)
        y_ = np_to_tf_one_hot(y_)

        _, h_prev_ = sess.run([optimizer, h_prev],
                              feed_dict={x: x_,
                                         y: y_,
                                         h: h_prev_})

        # sample
        if n % print_step == 0:
            sample_sequence = []
            for t in range(200):
                p_s_, h_prev_ = sess.run([p_s, h_prev],
                                            feed_dict={x: x_,
                                                       y: y_,
                                                       h_prev: h_prev_})
                y_next = np.random.choice(range(vocabulary_size),
                                          p=p_s_.ravel())
                sample_sequence.append(y_next)

            txt = ''.join(index_to_char[ix] for ix in sample_sequence)
            print ('----\n {} \n----'.format(txt))

        # move data pointer
        p += sequence_length
        # iteration counter
        n += 1
