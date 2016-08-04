'''
Created on 30 July 2016
@author: Kolesnikov Sergey
'''
# raw version
from optparse import OptionParser
import tensorflow as tf
import numpy as np
from data_io import *


# hyperparameters
learning_rate   = 1e-1
sequence_length = 25 # number of step_s to unroll the RNN for
hidden_size     = 100 # size of hidden layer of neurons
max_steps = 100000
print_log_step = 1000
print_sample_step = 10000

model_filepath = None
CONTINUE_LEARNING = False

x = tf.placeholder(tf.float32, [None, vocabulary_size], name="x")
y = tf.placeholder(tf.float32, [None, vocabulary_size], name="y")
h = tf.placeholder(tf.float32, [1, hidden_size], name="h")

initializer = tf.random_normal_initializer(stddev=0.1)

# model parameters
with tf.variable_scope("rnn", reuse=False) as rnn_scope:
    # weights
    # input to hidden
    W_xh = tf.get_variable("W_xh", [vocabulary_size, hidden_size],
                           tf.float32, initializer=initializer)
    # hidden to hidden
    W_hh = tf.get_variable("W_hh", [hidden_size, hidden_size],
                           tf.float32, initializer=initializer)
    # hidden to output
    W_hy = tf.get_variable("W_hy", [hidden_size, vocabulary_size],
                           tf.float32, initializer=initializer)

    # biases
    # hidden bias
    b_h = tf.get_variable("b_h", [hidden_size],
                          tf.float32, initializer=initializer)
    # output bias
    b_y = tf.get_variable("b_y", [vocabulary_size],
                          tf.float32, initializer=initializer)

# rnn model
with tf.variable_scope("rnn", reuse=True) as rnn_scope:
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

def tf_one_hot(x):
    return np.eye(vocabulary_size)[x]

def rnn_run():
    # char-rnn model (aka forward pass)
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
    minimizer = tf.train.AdamOptimizer()
    grad_var = minimizer.compute_gradients(loss)

    # hack for preventing exploding gradients
    grad_var = [(tf.clip_by_value(grad, -5.0, 5.0), var) \
                    for grad, var in grad_var]

    # update
    update = minimizer.apply_gradients(grad_var)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        n, n_, p = 0, 0, 0

        if model_filepath and CONTINUE_LEARNING:
            saver.restore(sess, model_filepath)

        while n <= max_steps:
            if p + sequence_length + 1 >= len(data) or n == 0:
                h_prev_ = np.zeros([1, hidden_size])
                p = 0

            x_ = [char_to_index[ch] for ch in data[p:p+sequence_length]]
            y_ = [char_to_index[ch] for ch in data[p+1:p+sequence_length+1]]

            x_ = tf_one_hot(x_)
            y_ = tf_one_hot(y_)

            h_prev_, loss_, _ = sess.run([h_prev, loss, update],
                                         feed_dict={x: x_,
                                                    y: y_,
                                                    h: h_prev_})


            if n % print_log_step == 0:
                print ('iter: {},  loss: {}'.format(n, loss_))

            # sample
            if n % print_sample_step == 0:
                sample_sequence = []
                start_index = np.random.randint(0, len(data) - sequence_length)
                sample_indeces = [
                    char_to_index[ch] for ch in \
                        data[start_index:start_index + sequence_length]
                    ]
                h_prev_sample = np.copy(h_prev_)
                for t in range(200):
                    sample_x = tf_one_hot(sample_indeces)
                    p_s_, h_prev_sample = sess.run([p_s, h_prev],
                                                   feed_dict={x: sample_x,
                                                              h: h_prev_sample})
                    y_next = np.random.choice(range(vocabulary_size),
                                              p=p_s_.ravel())
                    sample_sequence.append(y_next)
                    sample_indeces = sample_indeces[1:] + [y_next]

                txt = ''.join(index_to_char[ix] for ix in sample_sequence)
                print ('----\n {} \n----'.format(txt))

            # move data pointer
            p += sequence_length
            # iteration counter
            n += 1
            n_ += 1

        if model_filepath:
            saver.save(sess, model_filepath)

if __name__ == "__main__":
    # prepare options parser
    parser = OptionParser(usage='%prog [options]',
                          description='Simple numpy char-rnn')
    parser.add_option('-f', '--model-filepath',
                      dest='model_filepath',
                      help='filepath for model save/load',
                      default=None)
    parser.add_option('-c', '--continue-learning',
                      dest='continue_learning',
                      help='flag to continue learning the loaded model',
                      default=False)

    options, args = parser.parse_args()

    model_filepath = options.model_filepath
    CONTINUE_LEARNING = options.continue_learning

    rnn_run()
