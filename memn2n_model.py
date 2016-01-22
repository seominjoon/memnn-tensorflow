import tensorflow as tf

from data import DataSet


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


class MemN2NModel(object):
    def __init__(self, config, session):
        self.config = config
        self.session = session

        # place holders
        self.x_batch = tf.placeholder('float', name='x', shape=[None, config.memory_size, config.sentence_size])
        self.q_batch = tf.placeholder('float', name='q', shape=[None, config.sentence_size])
        self.a_batch = tf.placeholder('float', name='a', shape=[None, config.vocab_size])
        self.y_batch = tf.placeholder('float', name='y', shape=[None, config.vocab_size])
        self.learning_rate = tf.placeholder('float', name='lr')

        # input embedding
        self.A_ems, self.C_ems = [], []
        for i in range(config.num_layer):
            if i == 0:
                A_em = EmbeddingModule(config, A_name='A%d' % i, TA_name='TA%d' % i)
                C_em = EmbeddingModule(config, A_name='C%d' % i, TA_name='TC%d' % i)
            else:
                if config.tying == 'adj':
                    A_em = EmbeddingModule(config, C_em)
                    C_em = EmbeddingModule(config, A_name='C%d' % i, TA_name='TC%d' % i)
                elif config.tying == 'rnn':
                    A_em = EmbeddingModule(config, A_em)
                    C_em = EmbeddingModule(config, C_em)
                else:
                    raise Exception("undefined tying method")
            self.A_ems.append(A_em)
            self.C_ems.append(C_em)

        # question embedding
        if config.tying == 'adj':
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

        # connect tensors
        u_batch = self.B_em(self.q_batch)
        for i, (A_em, C_em, memory_layer) in enumerate(zip(self.A_ems, self.C_ems, self.memory_layers)):
            o_batch = memory_layer(A_em(self.x_batch), C_em(self.x_batch), u_batch)
            if config.tying == 'rnn':
                u_batch = tf.matmul(u_batch, self.H)
            u_batch = u_batch + o_batch

        # output tensor
        self.a_batch = tf.nn.softmax(tf.matmul(u_batch, self.W))

        # accuracy tensor
        correct_prediction = tf.equal(tf.argmax(self.a_batch, 1), tf.argmax(self.y_batch, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        # loss tensor
        self.loss = -tf.reduce_sum(self.y_batch * tf.log(self.a_batch))

        # optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, x_batch, q_batch, y_batch, learning_rate):
        self.optimizer.run(feed_dict={self.x_batch: x_batch, self.q_batch: q_batch, self.y_batch: y_batch,
                                      self.learning_rate: learning_rate})

    def train_data_set(self, train_data_set, val_data_set=None, val_period=1):
        """
        :param train_data_set:
        :param val_data_set: If val_data_set is specified, then intermediate results are printed for every val_period epochs
        :param val_period:
        :return:
        """
        assert isinstance(train_data_set, DataSet)
        num_epoch = self.config.num_epoch
        batch_size = self.config.batch_size
        lr, anneal_ratio, anneal_period = self.config.init_lr, self.config.anneal_ratio, self.config.anneal_period
        for epoch_idx in range(num_epoch):
            if epoch_idx > 0 and epoch_idx % anneal_period == 0:
                lr *= anneal_ratio
            while train_data_set.has_next(batch_size):
                x_batch, q_batch, y_batch = train_data_set.next_batch(batch_size)
                self.train(x_batch, q_batch, y_batch, lr)
            train_data_set.rewind()
            if val_data_set is not None and epoch_idx % val_period == 0:
                print self.test_data_set(val_data_set)

    def test(self, x_batch, q_batch, y_batch):
        return self.accuracy.eval(feed_dict={self.x_batch: x_batch, self.q_batch: q_batch, self.y_batch: y_batch})

    def test_data_set(self, data_set):
        assert isinstance(data_set, DataSet)
        return self.test(data_set.xs, data_set.qs, data_set.ys)