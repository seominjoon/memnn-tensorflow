import tensorflow as tf


class MemoryLayer(object):
    def __init__(self, config, A=None, B=None, C=None, layer_name=None):
        self._hidden_size = config.hidden_size  # d
        self._vocab_size = config.vocab_size  # V
        self._memory_size = config.memory_size
        if A is None:
            self.A = tf.get_variable('A', shape=[self._vocab_size, self._hidden_size])
        else:
            self.A = A
        if B is None:
            self.B = tf.get_variable('B', shape=[self._vocab_size, self._hidden_size])
        else:
            self.B = B
        if C is None:
            self.C = tf.get_variable('C', shape=[self._vocab_size, self._hidden_size])
        else:
            self.C = C

    def __call__(self, x_batch, q_batch, sentence_sizes_batch):
        """

        :param x_batch: [B, M, J, V]
        :param q_batch: [B, J, V]
        :param sentence_sizes_batch:
        :return:
        """
        xA_batch = tf.matmul(x_batch, self.A)  # [B, M, J, d]
        xC_batch = tf.matmul(x_batch, self.C)  # [B, M, J, d]
        xB_batch = tf.matmul(q_batch, self.B)  # [B, J, d]
        l_batch = self._pe_batch(sentence_sizes_batch)
        m_batch = tf.reduce_sum(tf.mul(xA_batch, l_batch), [2])  # [B, M, d]
        c_batch = tf.reduce_sum(tf.mul(xC_batch, l_batch), [2])
        u_batch = tf.reduce_sum(tf.mul(xB_batch, l_batch), [2])




    def _pe(self, sentence_size):
        f = lambda J, j, d, k: (1-j/J) - (k/d)*(1-2*j/J)
        g = lambda j: tf.concat(0, [f(sentence_size, j, self._hidden_size, k) for k in range(self._hidden_size)])
        l = tf.concat(0, [g(j) for j in range(sentence_size)])
        return l

    def _pe_batch(self, sentence_sizes_batch):
        # [B, M], [J,d] -> [B, M, J, d]
        l_batch = tf.concat(0, [tf.concat(0, [self._pe(sentence_size) for sentence_size in sentence_sizes])
                                for sentence_sizes in sentence_sizes_batch])
        return l_batch
