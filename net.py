import numpy as np
import matplotlib.pyplot as plt
import IPython
from numba import jit
import os
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf


class Network():

    def __init__(self, arch, learning_rate = .01, epochs = 400):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.arch = np.array(arch)
        self.learning_rate = learning_rate
        self.constructed = False
        self.bsize = 200
        self.iters = epochs * 10
        self.mean = None
        self.std = None

    @jit
    def whiten(self, X):
        X = X - self.mean
        X = X / self.std
        locs = np.isnan(X)
        X[locs] = 0.0
        locs = np.isinf(X)
        X[locs] = 0.0
        return X

    @jit
    def params(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)


    def predict(self, X):
        X = np.array(X)
        X = self.whiten(X)

        with self.graph.as_default():
            with self.sess.as_default():
                fd = {self.states: X}
                preds = self.sess.run(self.outputs, feed_dict = fd)
        return preds

    def construct(self, n, d):
        with self.graph.as_default():
            with self.sess.as_default():

                arch = np.insert(self.arch, 0, n)

                self.states = tf.placeholder("float", [None, n])
                self.actions = tf.placeholder("float", [None, d])

                buffer_layer = self.states
                # if self.mean is not None and self.std is not None:
                #     buffer_layer = (buffer_layer - self.mean) / (self.std + 1e-6)

                layers = [buffer_layer]
                last_layer = buffer_layer

                weights = []
                biases = []

                for i in range(len(arch))[1:]:
                    size = arch[i]
                    prev_size = arch[i - 1]
                    w1 = tf.Variable(tf.random_normal([prev_size, size], stddev=.15), name='w' + str(i))
                    b1 = tf.Variable(tf.random_normal([size], stddev=.15), name='b' + str(i))
                    weights.append(w1)
                    biases.append(b1)
                    layer = tf.nn.tanh(tf.matmul(layers[i-1], w1) + b1)
                    layers.append(layer)

                w1 = tf.Variable(tf.random_normal([arch[-1], d], stddev=.15), name='w' + str(len(arch) - 1))
                b1 = tf.Variable(tf.random_normal([d], stddev=.15), name='b' + str(len(arch) - 1))
                weights.append(w1)
                biases.append(b1)
                self.weights = weights
                self.biases = biases

                self.outputs = tf.matmul(layers[-1], w1) + b1

                self.loss = tf.reduce_mean(tf.square(self.outputs - self.actions))
                self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

                self.sum_loss = tf.reduce_sum(tf.square(self.outputs - self.actions))
                self.r_squared = 1 - self.sum_loss / (tf.reduce_sum(tf.square(self.actions - tf.reduce_mean(self.actions, 0))))

                self.sess.run(tf.global_variables_initializer())



    def load_weights(self, weights_file, stats_file):
        weights = pickle.load(open(weights_file, 'r'))[:-1]
        stats = pickle.load(open(stats_file, 'r'))

        n = weights[0].shape[0]
        d = weights[-1].shape[0]

        self.construct(n, d)
        with self.graph.as_default():
            with self.sess.as_default():
                for i in range(len(self.weights)):
                    w1 = self.weights[i]
                    b1 = self.biases[i]
                    w2 = weights[2 * i]
                    b2 = weights[2 * i + 1]

                    op1 = w1.assign(w2)
                    op2 = b1.assign(b2)

                    op1.eval()
                    op2.eval()

        self.mean, self.std = stats



 