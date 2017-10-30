import theano
import theano.tensor as T
import numpy as np
import sys
import time

MIN_LENGTH = 50
MAX_LENGTH = 55
hidden_layer = 100
the_input = 2
the_output = 1


def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH):
    # generate x_seq
    length = np.random.randint(min_length, max_length)
    x_seq = np.concatenate([np.random.uniform(size=(length, 1)), np.zeros((length, 1))], axis=-1)

    # set the second dimension to 1 at the indices to add
    x_seq[np.random.randint(length / 10), 1] = 1
    x_seq[np.random.randint(length / 2, length), 1] = 1  # np.random.randint(low,high)

    # multiply and sum the dimensions of x_seq to get the target value
    y_hat = np.sum(x_seq[:, 0] * x_seq[:, 1])
    return x_seq, y_hat


# print(gen_data())


x_seq = T.matrix('input')
y_hat = T.scalar('target')

wi = theano.shared(np.random.randn(the_input, the_output))
bh = theano.shared(np.zeros(hidden_layer))
wo = theano.shared(np.random.randn(hidden_layer, the_output))
bo = theano.shared(np.zeros(the_output))
wh = theano.shared(np.random.randn(hidden_layer, hidden_layer))
parameters = [wi, bh, wo, bo, wh]


def sigmoid(z):
    return 1 / (1 + T.exp(-z))


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     return np.exp(x) / np.sum(np.exp(x), axis=0)


def step(x_t, a_tm1, y_tm1):
    a_t = sigmoid(T.dot(x_t, wi) + T.dot(a_tm1, wh) + bh)
    # y_t = softmax(T.dot(a_t, wo) + bo)
    y_t = T.dot(a_t, wo) + bo
    return a_t, y_t


a_0 = theano.shared(np.zeros(hidden_layer))
y_0 = theano.shared(np.zeros(outputs))

[a_seq, y_seq], _ = theano.scan(step, sequences= x_seq, outputs_info= [a_0, y_0], truncate_gradient=-1)

y_seq_last = y_seq[-1][0]
cost = T.sum((y_seq_last - y_hat)**2)
gradient = T.grad(cost, parameters)


def para_update(parameters, gradient):
    mu = np.float32(0.001)
    parameters_updates = [(p, p-mu*g) for p,g in zip(parameters, gradient)]
    return parameters_updates


rnn_test = theano.function(inputs= [x_seq], outputs=y_seq_last)
rnn_train = theano.function(inputs=[x_seq, y_hat], outputs=cost, updates=para_update(parameters, gradient))

for i in range(10):
    x_seq, y_hat = gen_data()
    print("reference", y_hat, "RNN output:", rnn_test(x_seq))

for i in range(10000000):
    x_seq, y_hat = gen_data()
    print("iteration:", i, "cost:", rnn_train(x_seq, y_hat))
