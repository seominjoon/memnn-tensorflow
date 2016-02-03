import tensorflow as tf
import numpy as np

from data import DataSet


class Model(object):
    def __init__(self, graph, params, log_dir=None, gpu=False):
        self.graph = graph
        self.params = params
        self.gpu = gpu
        with graph.as_default():
            with tf.device('/cpu:0'):
                self.variables = self._init_variables()
            self.train_tensors = self._build_graph('train')
            self.linear_train_tensors = self._build_graph('train', linear=True)
            self.val_tensors = self._build_graph('val')
            self.test_tensors = self._build_graph('test')
            if log_dir is not None:
                self.writer = tf.train.SummaryWriter(log_dir, graph.as_graph_def())
            else:
                self.writer = None

    def _init_variables(self):
        params = self.params
        V, M, d = params.vocab_size, params.memory_size, params.hidden_size

        class Variables(object):
            pass

        variables = Variables()
        global_step = tf.Variable(0, trainable=False, name='global_step')
        variables.global_step = global_step
        with tf.variable_scope("var", initializer=tf.random_normal_initializer(params.init_mean, params.init_std)):
            As, TAs, Cs, TCs = [], [], [], []
            for layer_index in xrange(params.num_layers):
                with tf.variable_scope("layer_%d" % layer_index):
                    if layer_index == 0:
                        A = tf.get_variable('A', dtype='float', shape=[V, d])
                        TA = tf.get_variable('TA', dtype='float', shape=[M, d])
                        C = tf.get_variable('C', dtype='float', shape=[V, d])
                        TC = tf.get_variable('TC', dtype='float', shape=[M, d])
                    else:
                        if params.tying == 'adj':
                            A = tf.identity(Cs[-1], name='A')
                            TA = tf.identity(TCs[-1], name='TA')
                            C = tf.get_variable('C', dtype='float', shape=[V, d])
                            TC = tf.get_variable('TC', dtype='float', shape=[M, d])
                        elif params.tying == 'rnn':
                            A = tf.identity(As[-1], name='A')
                            TA = tf.identity(TAs[-1], name='TA')
                            C = tf.identity(Cs[-1], name='C')
                            TC = tf.identity(TCs[-1], name='TC')
                        else:
                            raise Exception('Unknown tying method: %s' % params.tying)
                    As.append(A)
                    TAs.append(TA)
                    Cs.append(C)
                    TCs.append(TC)

            if params.tying == 'adj':
                B = tf.identity(As[0], name='B')
                W = tf.transpose(Cs[-1], name='W')
            elif params.tying == 'rnn':
                B = tf.get_variable('B', dtype='float', shape=[V, d])
                W = tf.get_variable('W', dtype='float', shape=[d, V])
            else:
                raise Exception('Unknown tying method: %s' % params.tying)

            if params.tying == 'rnn':
                H = tf.get_variable('H', dtype='float', shape=[d, d])
                variables.H = H

            variables.As, variables.TAs, variables.B, variables.Cs, variables.TCs, variables.W = As, TAs, B, Cs, TCs, W

        return variables

    def _get_l(self):
        J, d = self.params.max_sentence_size, self.params.hidden_size
        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)
        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]
        l = [g(j) for j in range(J)]
        l_tensor = tf.constant(l, shape=[J, d], name='l')
        return l_tensor

    def _softmax_with_mask(self, um_batch, m_mask_batch):
        exp_um_batch = tf.exp(um_batch)  # [N, M]
        masked_batch = exp_um_batch * m_mask_batch  # [N, M]
        sum_2d_batch = tf.expand_dims(tf.reduce_sum(masked_batch, 1), -1)  # [N, 1]
        p_batch = tf.div(masked_batch, sum_2d_batch, name='p')  # [N, M]
        return p_batch

    def _build_graph(self, mode, linear=False):
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

        As, TAs, B, Cs, TCs, W = variables.As, variables.TAs, variables.B, variables.Cs, variables.TCs, variables.W
        if params.tying == 'rnn':
            H = variables.H

        class Tensors(object):
            pass
        tensors = Tensors()

        summaries = []

        # initialize tensors
        with tf.variable_scope(mode):
            # placeholders
            with tf.name_scope('ph'):
                with tf.name_scope('x'):
                    x_batch = tf.placeholder('int32', shape=[N, M, J], name='x')
                    x_mask_batch = tf.placeholder('float', shape=[N, M, J], name='x_mask')
                    x_mask_aug_batch = tf.expand_dims(x_mask_batch, -1, 'x_mask_aug')
                    m_mask_batch = tf.placeholder('float', shape=[N, M], name='m_mask')
                    tensors.x = x_batch
                    tensors.x_mask = x_mask_batch
                    tensors.m_mask = m_mask_batch

                with tf.name_scope('q'):
                    q_batch = tf.placeholder('int32', shape=[N, J], name='q')
                    q_mask_batch = tf.placeholder('float', shape=[N, J], name='q_mask')
                    q_mask_aug_batch = tf.expand_dims(q_mask_batch, -1, 'q_mask_aug')
                    tensors.q = q_batch
                    tensors.q_mask = q_mask_batch

                y_batch = tf.placeholder('int32', shape=[N], name='y')
                tensors.y = y_batch

                learning_rate = tf.placeholder('float', name='lr')
                tensors.learning_rate = learning_rate

            with tf.name_scope('const'):
                l = self._get_l()  # [J, d]
                l_aug = tf.expand_dims(l, 0, name='l_aug')
                l_aug_aug = tf.expand_dims(l_aug, 0, name='l_aug_aug')  # [1, 1, J, d]

            with tf.name_scope('a'):
                a_batch = tf.nn.embedding_lookup(tf.diag(tf.ones(shape=[V])), y_batch, name='a')  # [N, d]

            with tf.name_scope('first_u'):
                Bq_batch = tf.nn.embedding_lookup(B, q_batch)  # [N, J, d]
                if params.position_encoding:
                    Bq_batch *= l_aug
                Bq_batch *= q_mask_aug_batch
                first_u_batch = tf.reduce_sum(Bq_batch, 1, name='first_u')  # [N, d]

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
                        if params.position_encoding:
                            Ax_batch *= l_aug_aug  # position encoding
                        Ax_batch *= x_mask_aug_batch  # masking
                        m_batch = tf.reduce_sum(Ax_batch, 2)  # [N, M, d]
                        m_batch = tf.add(tf.expand_dims(TAs[layer_index], 0), m_batch, name='m')  # temporal encoding

                    with tf.name_scope('c'):
                        Cx_batch = tf.nn.embedding_lookup(Cs[layer_index], x_batch)  # [N, M, J, d]
                        if params.position_encoding:
                            Cx_batch *= l_aug_aug  # position encoding
                        Cx_batch *= x_mask_aug_batch
                        c_batch = tf.reduce_sum(Cx_batch, 2)
                        c_batch = tf.add(tf.expand_dims(TCs[layer_index], 0), c_batch, name='c')  # temporal encoding

                    with tf.name_scope('p'):
                        u_batch_aug = tf.expand_dims(u_batch, -1)  # [N, d, 1]
                        um_batch = tf.squeeze(tf.batch_matmul(m_batch, u_batch_aug), [2])  # [N, M]
                        if linear:
                            p_batch = tf.mul(um_batch, m_mask_batch, name='p')
                        else:
                            p_batch = self._softmax_with_mask(um_batch, m_mask_batch)

                    with tf.name_scope('o'):
                        o_batch = tf.reduce_sum(c_batch * tf.expand_dims(p_batch, -1), 1)  # [N, d]


                u_batch_list.append(u_batch)
                o_batch_list.append(o_batch)

            if params.tying == 'rnn':
                last_u_batch = tf.add(tf.matmul(u_batch_list[-1], H), o_batch_list[-1], name='last_u')
            else:
                last_u_batch = tf.add(u_batch_list[-1], o_batch_list[-1], name='last_u')

            with tf.name_scope('ap'):
                ap_raw_batch = tf.matmul(last_u_batch, W, name='ap_raw')  # [N d] X [d V] = [N V]
                ap_batch = tf.nn.softmax(ap_raw_batch, name='ap')

            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(ap_raw_batch, a_batch)
                avg_loss = tf.reduce_mean(losses, 0)
                summaries.append(tf.scalar_summary('avg_loss', avg_loss))
                tensors.losses = losses
                tensors.avg_loss = avg_loss

            with tf.name_scope('acc'):
                correct = tf.equal(tf.argmax(ap_batch, 1), tf.argmax(a_batch, 1))
                acc = tf.reduce_mean(tf.cast(correct, 'float'), name='acc')
                tensors.acc = acc

            if mode == 'train':
                opt = tf.train.GradientDescentOptimizer(learning_rate)
                grads_and_vars = opt.compute_gradients(losses)
                clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
                opt_op = opt.apply_gradients(clipped_grads_and_vars, global_step=self.variables.global_step)
                tensors.opt_op = opt_op

        tensors.summary = tf.merge_summary(summaries)

        return tensors

    def _get_feed_dict(self, tensors, x_raw, q_raw, y):
        (x, x_mask, m_mask), (q, q_mask) = self._preprocess(x_raw, q_raw)
        feed_dict = {tensors.x: x, tensors.x_mask: x_mask, tensors.m_mask: m_mask,
                     tensors.q: q, tensors.q_mask: q_mask, tensors.y: y}
        return feed_dict

    def train_batch(self, sess, tensors, x, q, y, learning_rate, eval_tensors=None):
        # Need to initialize all variables in sess before training!
        feed_dict = self._get_feed_dict(tensors, x, q, y)
        feed_dict[tensors.learning_rate] = learning_rate
        sess.run(tensors.opt_op, feed_dict=feed_dict)
        if eval_tensors is not None:
            return sess.run(eval_tensors, feed_dict)
        else:
            return None

    def train(self, sess, train_data_set, val_data_set, eval_period=1):
        assert isinstance(train_data_set, DataSet)
        assert isinstance(val_data_set, DataSet)
        params = self.params
        num_epochs = params.num_epochs
        batch_size = params.train_batch_size
        learning_rate = params.init_lr
        linear = params.linear_start
        if linear:
            print "Starting with linear learning."
        for epoch_idx in xrange(num_epochs):
            tensors = self.linear_train_tensors if linear else self.train_tensors
            while train_data_set.has_next(batch_size):
                # global_idx = epoch_idx * (train_data_set.num_examples / batch_size) + train_data_set._index_in_epoch
                x, q, y = train_data_set.next_batch(batch_size)
                eval_tensors = [tensors.summary, self.variables.global_step, tensors.acc]
                result = self.train_batch(sess, tensors, x, q, y, learning_rate, eval_tensors=eval_tensors)
                summary_str, global_step, train_acc = result
                if self.writer is not None:
                    self.writer.add_summary(summary_str, global_step)
            train_data_set.rewind()

            val_avg_loss, acc = self.test(sess, val_data_set, 'val')
            if epoch_idx > 0 and epoch_idx % eval_period == 0:
                print "iter %d: train_err=%.2f%%, val_err=%.2f%%, val_avg_loss=%.3f, lr=%f" % \
                      (epoch_idx, (1-train_acc)*100, (1-acc)*100, val_avg_loss, learning_rate)
            if epoch_idx > 0 and epoch_idx % params.anneal_period == 0:
                learning_rate *= params.anneal_ratio
            if linear and epoch_idx >= 20:
                print "Linear learning ended."
                linear = False

    def test(self, sess, test_data_set, mode):
        x, q, y = test_data_set.xs, test_data_set.qs, test_data_set.ys

        if mode == 'val':
            tensors = self.val_tensors
        elif mode == 'test':
            tensors = self.test_tensors
        else:
            raise Exception()

        feed_dict = self._get_feed_dict(tensors, x, q, y)
        loss, acc = sess.run([tensors.avg_loss, tensors.acc], feed_dict=feed_dict)
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
