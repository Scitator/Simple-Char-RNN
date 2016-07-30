'''
Created on 29 July 2016
@author: Kolesnikov Sergey
'''
# raw version
import numpy as np
from data_io import *


# hyperparameters
learning_rate   = 1e-1
sequence_length = 25 # number of step_s to unroll the RNN for
hidden_size     = 100 # size of hidden layer of neurons
max_steps = 1000000
print_step = 1000

# model parameters

# weights
# input to hidden
W_xh = np.random.randn(hidden_size, vocabulary_size)*0.01
# hidden to hidden
W_hh = np.random.randn(hidden_size, hidden_size)*0.01
# hidden to output
W_hy = np.random.randn(vocabulary_size, hidden_size)*0.01

# biases
# hidden bias
b_h = np.zeros((hidden_size, 1))
# output bias
b_y = np.zeros((vocabulary_size, 1))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def rnn_step(x : list,
             y : list,
             h_prev : np.array):
    """
    x, y - list of integers,
    h_prev - Hx1 array of previous hidden state.

    returns loss, gradients, and previous hidden state
    """
    x_s, h_s, y_s, p_s = {}, {}, {}, {}
    h_s[-1] = np.copy(h_prev)
    loss = 0

    # forward pass
    for t in range(len(x)):
        # encode in 1-of-k representation
        x_s[t] = np.zeros((vocabulary_size, 1))
        x_s[t][x[t]] = 1
        # compute rnn-cell hidden state
        h_s[t] = np.tanh(np.dot(W_xh, x_s[t]) + np.dot(W_hh, h_s[t-1]) + b_h)
        # unnormalized log probabilities for next chars
        y_s[t] = np.dot(W_hy, h_s[t]) + b_y
        # probabilities for next chars# hidden to output
        p_s[t] = softmax(y_s[t])
        # softmax cross-entropy loss
        loss += -np.log(p_s[t][y[t], 0])

    # backward pass
    dW_xh, dW_hh, dW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
    db_h, db_y = np.zeros_like(b_h), np.zeros_like(b_y)
    dh_next = np.zeros_like(h_s[0])

    for t in reversed(range(len(x))):
        dy = np.copy(p_s[t])
        dy[y[t]] -= 1

        dW_hy += np.dot(dy, h_s[t].T)
        db_y += dy

        # backprop into h
        dh = np.dot(W_hy.T, dy) + dh_next
        # backprop through tanh nonlinearity
        dh_raw = (1 - h_s[t] * h_s[t]) * dh
        db_h += dh_raw

        dW_xh += np.dot(dh_raw, x_s[t].T)
        dW_hh += np.dot(dh_raw, h_s[t-1].T)
        dh_next = np.dot(W_hh.T, dh_raw)

    # hack for preventing exploding gradients
    for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dW_xh, dW_hh, dW_hy, db_h, db_y, h_s[len(x)-1]

def rnn_sample(h, index_seed, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocabulary_size, 1))
    x[index_seed] = 1
    indices = []
    for t in range(n):
        h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + b_h)
        y = np.dot(W_hy, h) + b_y
        p = softmax(y)
        ix = np.random.choice(range(vocabulary_size), p=p.ravel())
        x = np.zeros((vocabulary_size, 1))
        x[ix] = 1
        indices.append(ix)
    return indices

def rnn_test():
    n, p = 0, 0
    # memory variables for Adagrad
    mW_xh, mW_hh, mW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
    mb_h, mb_y = np.zeros_like(b_h), np.zeros_like(b_y)

    # loss at iteration 0
    smooth_loss = -np.log(1.0/vocabulary_size) * sequence_length

    while n <= max_steps:
        # prepare x (we're sweeping from left to right in steps sequence_length long)
        if p + sequence_length + 1 >= len(data) or n == 0:
            # reset RNN memory
            h_prev = np.zeros((hidden_size,1))
            # go from start of data
            p = 0

        x = [char_to_index[ch] for ch in data[p:p+sequence_length]]
        y = [char_to_index[ch] for ch in data[p+1:p+sequence_length+1]]

        # sample from the model
        if n % print_step == 0:
            sample_ix = rnn_sample(h_prev, x[0], 200)
            txt = ''.join(index_to_char[ix] for ix in sample_ix)
            print ('----\n {} \n----'.format(txt))

        # forward sequence_length characters through the network and fetch gradient
        loss, dW_xh, dW_hh, dW_hy, db_h, db_y, h_prev = rnn_step(x, y, h_prev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # print progress
        if n % print_step == 0:
            print ('iter {}, loss: {}'.format(n, smooth_loss))

        # parameter update with Adagrad
        for param, dparam, mem in zip([W_xh, W_hh, W_hy, b_h, b_y],
                                      [dW_xh, dW_hh, dW_hy, db_h, db_y],
                                      [mW_xh, mW_hh, mW_hy, mb_h, mb_y]):
            mem += dparam ** 2
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

        # move data pointer
        p += sequence_length
        # iteration counter
        n += 1

if __name__ == "__main__":
    rnn_test()
