import tensorflow as tf
import numpy as np

from data import DataSet


class Model(object):
    def __init__(self, graph, params, logdir=None):
        self.graph = graph
        with graph.as_default():
            self.params = params
            self.variables = self._init_variables()
            self.train_tensors = self._build_graph('train')
            self.val_tensors = self._build_graph('val')
            self.test_tensors = self._build_graph('test')

            if logdir is not None:
                self.writer = tf.train.SummaryWriter(logdir, graph.as_graph_def())
            else:
                self.writer = None

    def _init_variables(self):
        params = self.params
        V, d = params.vocab_size, params.hidden_size

        class Variables(object):
            pass

        variables = Variables()
        with tf.variable_scope("var", initializer=tf.random_normal_initializer(params.init_mean, params.init_std)):
            As, Cs = [], []
            for layer_index in xrange(params.num_layers):
                with tf.variable_scope("l%d" % layer_index):
                    if layer_index == 0:
                        A = tf.get_variable('A', dtype='float', shape=[V, d])
                        C = tf.get_variable('C', dtype='float', shape=[V, d])
                    else:
                        if params.tying == 'adj':
                            A = Cs[-1]
                            C = tf.get_variable('C', dtype='float', shape=[V, d])
                        elif params.tying == 'rnn':
                            A = As[-1]
                            C = Cs[-1]
                        else:
                            raise Exception('Unknown tying method: %s' % params.tying)
                    As.append(A)
                    Cs.append(C)

            if params.tying == 'adj':
                B = tf.identity(As[0], name='B')
                W = tf.transpose(Cs[-1], name='W')
            elif params.tying == 'rnn':
                B = tf.get_variable('B', dtype='float', shape=[V, d])
                W = tf.get_variable('W', dtype='float', shape=[d, V])
            else:
                raise Exception('Unknown tying method: %s' % params.tying)

            variables.As, variables.B, variables.Cs, variables.W = As, B, Cs, W

        return variables

    def _build_graph(self, mode):
        params = self.params
        variables = self.variables
        M, J, V, d = params.memory_size, params.max_sentence_size, params.vocab_size, params.hidden_size
        if mode == 'train':
            N = params.train_batch_size
        elif mode == 'val':
            N = params.val_data_size
        elif mode == 'test':
            N = params.test_data_size
        else:
            raise Exception("Invalid graph mode: %s" % mode)

        learning_rate = params.init_lr

        As, B, Cs, W = variables.As, variables.B, variables.Cs, variables.W

        class Tensors(object):
            pass
        tensors = Tensors()

        summaries = []

        # initialize tensors
        with tf.variable_scope(mode):
            # placeholders
            with tf.name_scope('ph'):
                x_batch = tf.placeholder('int32', shape=[N, M, J], name='x')
                x_mask_batch = tf.placeholder('float', shape=[N, M, J], name='x_mask')
                q_batch = tf.placeholder('int32', shape=[N, J], name='q')
                q_mask_batch = tf.placeholder('float', shape=[N, J], name='q_mask')
                y_batch = tf.placeholder('int32', shape=[N], name='y')
                m_mask_batch = tf.placeholder('float', shape=[N, M], name='m_mask')

                tensors.x = x_batch
                tensors.x_mask = x_mask_batch
                tensors.q = q_batch
                tensors.q_mask = q_mask_batch
                tensors.y = y_batch
                tensors.m_mask = m_mask_batch

            with tf.name_scope('a'):
                a_batch = tf.nn.embedding_lookup(tf.diag(tf.ones(shape=[V])), y_batch, name='a')  # [N, d]

            with tf.name_scope('first_u'):
                Bq_batch = tf.nn.embedding_lookup(B, q_batch)  # [N, J, d]
                first_u_batch = tf.reduce_sum(tf.expand_dims(q_mask_batch, -1) * Bq_batch, 1, name='first_u')  # [N, d]

            u_batch_list = []
            o_batch_list = []
            for layer_index in xrange(params.num_layers):
                with tf.name_scope('layer_%d' % layer_index):
                    if layer_index == 0:
                        u_batch = tf.identity(first_u_batch, name='u')
                    else:
                        u_batch = tf.add(u_batch_list[-1], o_batch_list[-1], name='u')

                    with tf.name_scope('m'):
                        Ax_batch = tf.nn.embedding_lookup(As[layer_index], x_batch)  # [N, M, J, d]
                        m_batch = tf.reduce_sum(tf.expand_dims(x_mask_batch, -1) * Ax_batch, 2, name='m')  # [N, M, d]

                    with tf.name_scope('c'):
                        Cx_batch = tf.nn.embedding_lookup(Cs[layer_index], x_batch)
                        c_batch = tf.reduce_sum(tf.expand_dims(x_mask_batch, -1) * Cx_batch, 2, name='c')

                    with tf.name_scope('p'):
                        u_batch_x = tf.expand_dims(u_batch, -1)  # [N, d, 1]
                        p_batch = tf.nn.softmax(tf.squeeze(
                                    tf.batch_matmul(tf.expand_dims(m_mask_batch, -1) * m_batch, u_batch_x),
                                    [2]), name='p')  # [N, M]

                    with tf.name_scope('o'):
                        o_batch = tf.reduce_sum(c_batch * tf.expand_dims(p_batch, -1), 1)  # [N, d]

                u_batch_list.append(u_batch)
                o_batch_list.append(o_batch)

            last_u_batch = tf.add(u_batch_list[-1], o_batch_list[-1], name='last_u')

            with tf.name_scope('ap'):
                ap_batch = tf.nn.softmax(tf.matmul(last_u_batch, W), name='ap')  # [N d] X [d V] = [N V]

            with tf.name_scope('loss'):
                loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(ap_batch, a_batch, name='loss'), 0)
                tensors.loss = loss
                summaries.append(tf.scalar_summary('loss', loss))

            with tf.name_scope('acc'):
                correct = tf.equal(tf.argmax(ap_batch, 1), tf.argmax(a_batch, 1))
                acc = tf.reduce_mean(tf.cast(correct, 'float'), name='acc')
                tensors.acc = acc
            if mode == 'train':
                opt = tf.train.GradientDescentOptimizer(learning_rate)
                opt_op = opt.minimize(loss)
                tensors.opt_op = opt_op

        tensors.summary = tf.merge_summary(summaries)

        return tensors

    def _get_feed_dict(self, tensors, x_raw, q_raw, y):
        (x, x_mask, m_mask), (q, q_mask) = self._preprocess(x_raw, q_raw)
        feed_dict = {tensors.x: x, tensors.x_mask: x_mask, tensors.m_mask: m_mask,
                     tensors.q: q, tensors.q_mask: q_mask, tensors.y: y}
        return feed_dict

    def train_batch(self, sess, x, q, y):
        # Need to initialize all variables in sess before training!
        tensors = self.train_tensors
        feed_dict = self._get_feed_dict(tensors, x, q, y)
        ops = [tensors.opt_op]
        if self.writer is not None:
            ops.append(self.train_tensors.summary)
        return sess.run(ops, feed_dict=feed_dict)

    def train(self, sess, train_data_set, val_data_set, eval_period=1):
        assert isinstance(train_data_set, DataSet)
        assert isinstance(val_data_set, DataSet)
        params = self.params
        num_epochs = params.num_epochs
        batch_size = params.train_batch_size
        for epoch_idx in xrange(num_epochs):
            while train_data_set.has_next(batch_size):
                global_idx = epoch_idx * (train_data_set.num_examples / batch_size) + train_data_set._index_in_epoch
                x, q, y = train_data_set.next_batch(batch_size)
                result = self.train_batch(sess, x, q, y)
                if self.writer is not None:
                    self.writer.add_summary(result[1], global_idx)

            train_data_set.rewind()
            if epoch_idx > 0 and epoch_idx % eval_period == 0:
                loss, acc = self.test(sess, val_data_set, 'val')
                print "iter %d: acc=%.2f%%, loss=%.0f" % (epoch_idx, acc*100, loss)

    def test(self, sess, test_data_set, mode):
        x, q, y = test_data_set.xs, test_data_set.qs, test_data_set.ys

        if mode == 'val':
            tensors = self.val_tensors
        elif mode == 'test':
            tensors = self.test_tensors
        else:
            raise Exception()

        feed_dict = self._get_feed_dict(tensors, x, q, y)
        loss, acc = sess.run([tensors.loss, tensors.acc], feed_dict=feed_dict)
        return loss, acc

    def _preprocess(self, x_raw_batch, q_raw_batch):
        params = self.params
        N, M, J = len(x_raw_batch), params.memory_size, params.max_sentence_size
        x_batch = np.zeros([N, M, J])
        q_batch = np.zeros([N, J])
        x_mask_batch = np.zeros([N, M, J])
        q_mask_batch = np.zeros([N, J])
        m_mask_batch = np.zeros([N, M])

        for n, i, j in np.ndindex(x_batch.shape):
            if i < len(x_raw_batch[n]) and j < len(x_raw_batch[n][-i-1]):
                x_batch[n, i, j] = x_raw_batch[n][-i-1][j]
                x_mask_batch[n, i, j] = 1

        for n, j in np.ndindex(q_batch.shape):
            if j < len(q_raw_batch[n]):
                q_batch[n, j] = q_raw_batch[n][j]
                q_mask_batch[n, j] = 1

        for n, i in np.ndindex(m_mask_batch.shape):
            if i < len(x_raw_batch[n]):
                m_mask_batch[n, i] = 1

        return (x_batch, x_mask_batch, m_mask_batch), (q_batch, q_mask_batch)
