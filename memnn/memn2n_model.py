import tensorflow as tf


class EmbeddingModule(object):
    l_dict = {}

    def __init__(self, config, em=None, A=None, A_name="", TA=None, TA_name=""):
        self.config = config
        M, J, d, V = config.memory_size, config.sentence_size, config.hidden_size, config.vocab_size
        if em is not None:
            A, TA = em.A, em.TA
        if A is None:
            A = tf.get_variable(A_name, shape=[V, d])
        if config.te:
            if TA is None:
                TA = tf.get_variable(TA_name, shape=[M, d])
        self.A, self.TA = A, TA

    def __call__(self, x_batch):
        Ax_batch = tf.nn.embedding_lookup(self.A, x_batch)  # [B, M, J, d] or [B, J, d]
        if self.config.pe:
            Ax_batch *= EmbeddingModule._get_pe(self.config.sentence_size, self.config.hidden_size)
        m_batch = tf.reduce_sum(Ax_batch, [-2])  # [B, M, d] or [B, d]
        if self.config.te:
            m_batch += self.TA
        return m_batch

    @staticmethod
    def _get_pe(sentence_size, hidden_size):
        key = (sentence_size, hidden_size)
        if key not in EmbeddingModule.l_dict:
            f = lambda J, j, d, k: (1-j/J) - (k/d)*(1-2*j/J)
            g = lambda j: tf.concat(0, [f(sentence_size, j, hidden_size, k) for k in range(hidden_size)])
            l = tf.concat(0, [g(j) for j in range(sentence_size)])
            EmbeddingModule.l_dict[key] = l
        return EmbeddingModule.l_dict[key]


class MemoryLayer(object):
    def __init__(self, config, layer_name=None):
        self.config = config

    def __call__(self, m_batch, c_batch, u_batch):
        u_2d_batch = tf.expand_dims(u_batch, -1)

        p_2d_batch = tf.nn.softmax(tf.batch_matmul(m_batch, u_2d_batch)) # [B, M, 1]
        p_tiled_batch = tf.tile(p_2d_batch, [1, 1, self.config.hidden_size])  # [B, M, d]

        o_batch = tf.reduce_sum(c_batch * p_tiled_batch, [1])  # [B d]
        return o_batch


class MemN2N(object):
    def __init__(self, config):
        self.config = config
        num_layer = config.num_layer
        tying = config.tying

        # input embedding
        self.A_ems, self.C_ems = [], []
        for i in range(num_layer):
            if i == 0:
                A_em = EmbeddingModule(config, A_name='A', TA_name='TA')
                C_em = EmbeddingModule(config, A_name='C', TA_name='TC')
            else:
                if tying == 'adj':
                    A_em = EmbeddingModule(config, C_em, A_name='A', TA_name='TA')
                    C_em = EmbeddingModule(config, A_name='C', TA_name='TC')
                elif tying == 'rnn':
                    A_em = EmbeddingModule(config, A_em, A_name='A', TA_name='TA')
                    C_em = EmbeddingModule(config, C_em, A_name='A', TA_name='TA')
                else:
                    raise Exception("undefined tying method")
            self.A_ems.append(A_em)
            self.C_ems.append(C_em)

        # question embedding
        if tying == 'adj':
            self.B_em = EmbeddingModule(config, self.A_ems[0], A_name='B')
        else:
            self.B_em = EmbeddingModule(config, A_name='B')

        # memory layers
        self.memory_layers = [MemoryLayer(config) for _ in range(num_layer)]
        
        # linear mapping
        if config.tying == 'rnn':
            self.H = tf.get_variable('H', shape=[config.hidden_size, config.hidden_size])

        # output mapping
        if config.tying == 'adj':
            self.W = tf.transpose(self.C_ems[-1].A)
        elif config.tying == 'rnn':
            self.W = tf.get_variable('W', shape=[config.hidden_size, config.vocab_size])

    def __call__(self, x_batch, q_batch):
        u_batch = self.B_em(q_batch)
        for i, (A_em, C_em, memory_layer) in enumerate(zip(self.A_ems, self.C_ems, self.memory_layers)):
            o_batch = memory_layer(A_em(x_batch), C_em(x_batch), u_batch)
            if self.config.tying == 'rnn':
                u_batch = tf.matmul(u_batch, self.H)
            u_batch = u_batch + o_batch

        a_batch = tf.nn.softmax(tf.matmul(u_batch, self.W))
        return a_batch

    def loss(self, x_batch, q_batch, a_batch):
        """
        Computes loss function given labels (a_batch)
        :param x_batch:
        :param q_batch:
        :param a_batch:
        :return:
        """
