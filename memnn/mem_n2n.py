import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import math
import numpy as np


class MemoryLayer2(object):
    def __init__(self, config):
        self._num_units = config.num_units # d

    def __call__(self, inputs, memories, memory_lengths, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            memory_vectors = []
            for memory, memory_length in zip(memories, memory_lengths):
                m = tf.zeros([self._num_units])
                for j, x in enumerate(memory):
                    Ax = tf.nn.embedding_lookup(self.A, x)
                    l = self._pe(memory_length, j)
                    m += tf.mul(Ax, l)
                memory_vectors.append(m)


    def _pe(self, memory_length, j):
        f = lambda J, j, d, k: (1-j/J) - (k/d)*(1-2*j/J)
        l = [f(memory_length, j, self._num_units, k) for k in range(self._num_units)]
        return l

class MemN2N(object):
    def __init__(self):
        x = tf.placeholder("float", )



class Encoder(object):
    def __init__(self, data_set, d):
        self.vocab = data_set.vocab
        self.V = len(self.vocab)
        # d is embedding dimension
        self.d = d

    def encode(self, sentence):
        X = np.array([[int(self.equals(x, v)) for x in sentence] for v in self.vocab])
        return X

    def equals(self, a, b):
        return a == b


class QuestionInputLayer(object):
    def __init__(self, dim, B=None):
        self.dim = dim
        if B is None:
            B = tf.Variable(tf.truncated_normal(dim, stddev=0.1))
        self.B = B

    def put(self, q):
        u = tf.matmul(self.B, q)
        return u


class MemoryLayer(object):
    def __init__(self, dim, A=None, C=None):
        self.dim = dim
        if A is None:
            A = tf.Variable(tf.truncated_normal(dim, stddev=0.1))
        if C is None:
            C = tf.Variable(tf.truncated_normal(dim, stddev=0.1))
        self.A = A
        self.C = C
        self.AX_list = []
        self.CX_list = []
        self.p_list = []
        self.o_list = []

    def put(self, X, u):
        AX = tf.matmul(self.A, X)
        CX = tf.matmul(self.C, X)
        p = tf.nn.softmax(tf.matmul(tf.transpose(AX), u)) # M^T . u
        o = tf.reduce_sum(tf.matmul(CX, tf.diag(p)), 1)
        self.AX_list.append(AX)
        self.CX_list.append(CX)
        self.p_list.append(p)
        self.o_list.append(o)
        return o


class SumLayer(object):
    def __init__(self, dim=None, H=None):
        self.dim = dim
        if dim is None:
            self.H = None
        else:
            if H is None:
                H = tf.Variable(tf.truncated_normal(dim, 0.1))
            self.H = H
        self.s_list = []

    def put(self, u, o):
        if self.dim is None:
            s = u + o
        else:
            s = tf.matmul(self.H, u) + o
        self.s_list.append(s)
        return s


class OutputLayer(object):
    def __init__(self, dim):
        self.dim = dim
        self.W = tf.Variable(tf.truncated_normal(dim, 0.1))

    def put(self, s):
        a = tf.nn.softmax(tf.matmul(self.W, s))
        return a
