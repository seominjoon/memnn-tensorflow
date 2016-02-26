import tensorflow as tf
import numpy as np
import progressbar as pb
import os

from read_data import DataSet
from models.base_model import BaseModel


class N2NModel(BaseModel):
    def _build_tower(self):
        self.variables = self._init_variables()
        self._build_graph(self.params.linear_start)

    def _init_variables(self):
        params = self.params
        V, M, d = params.vocab_size, params.memory_size, params.hidden_size

        class Variables(object):
            pass

        variables = Variables()
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
        J, d = self.params.max_sent_size, self.params.hidden_size
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

    def _build_graph(self, linear=False):
        params = self.params
        variables = self.variables

        N, M, J, V, d = params.batch_size, params.memory_size, params.max_sent_size, params.vocab_size, params.hidden_size
        Q = params.max_ques_size

        As, TAs, B, Cs, TCs, W = variables.As, variables.TAs, variables.B, variables.Cs, variables.TCs, variables.W
        if params.tying == 'rnn':
            H = variables.H

        summaries = []

        # initialize self
        # placeholders
        with tf.name_scope('ph'):
            with tf.name_scope('x'):
                x_batch = tf.placeholder('int32', shape=[N, M, J], name='x')
                x_mask_batch = tf.placeholder('float', shape=[N, M, J], name='x_mask')
                x_mask_aug_batch = tf.expand_dims(x_mask_batch, -1, 'x_mask_aug')
                m_mask_batch = tf.placeholder('float', shape=[N, M], name='m_mask')
                self.x = x_batch
                self.x_mask = x_mask_batch
                self.m_mask = m_mask_batch

            with tf.name_scope('q'):
                q_batch = tf.placeholder('int32', shape=[N, J], name='q')
                q_mask_batch = tf.placeholder('float', shape=[N, J], name='q_mask')
                q_mask_aug_batch = tf.expand_dims(q_mask_batch, -1, 'q_mask_aug')
                self.q = q_batch
                self.q_mask = q_mask_batch

            y_batch = tf.placeholder('int32', shape=[N], name='y')
            self.y = y_batch

            learning_rate = tf.placeholder('float', name='lr')
            self.learning_rate = learning_rate

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
            logit_batch = tf.matmul(last_u_batch, W, name='ap_raw')  # [N d] X [d V] = [N V]
            ap_batch = tf.nn.softmax(logit_batch, name='ap')

        with tf.name_scope('loss') as loss_scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_batch, a_batch, name='cross_entropy')
            avg_cross_entropy = tf.reduce_mean(cross_entropy, 0, name='avg_cross_entropy')
            tf.add_to_collection('losses', avg_cross_entropy)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            losses = tf.get_collection('losses', loss_scope)
            self.total_loss = total_loss

        with tf.name_scope('acc'):
            correct_vec = tf.equal(tf.argmax(ap_batch, 1), tf.argmax(a_batch, 1))
            num_corrects = tf.reduce_sum(tf.cast(correct_vec, 'float'), name='num_corrects')
            acc = tf.reduce_mean(tf.cast(correct_vec, 'float'), name='acc')
            self.correct_vec = correct_vec
            self.num_corrects = num_corrects
            self.acc = acc

        with tf.name_scope('opt'):
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = opt.compute_gradients(cross_entropy)
            clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
            opt_op = opt.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)
            self.opt_op = opt_op

        summaries.append(tf.scalar_summary("%s (raw)" % total_loss.op.name, total_loss))
        self.merged_summary = tf.merge_summary(summaries)


    def _get_feed_dict(self, batch):
        sent_batch, ques_batch = batch[:2]
        if len(batch) > 2:
            label_batch = batch[2]
        else:
            label_batch = np.zeros([len(sent_batch)])
        x_batch, x_mask_batch, m_mask_batch = self._prepro_sent_batch(sent_batch)
        q_batch, q_mask_batch = self._prepro_ques_batch(ques_batch)
        y_batch = self._prepro_label_batch(label_batch)
        feed_dict = {self.x: x_batch, self.x_mask: x_mask_batch, self.m_mask: m_mask_batch,
                     self.q: q_batch, self.q_mask: q_mask_batch, self.y: y_batch}
        return feed_dict

    def _pad(self, array, inc):
        assert len(array.shape) > 0, "Array must be at least 1D!"
        if len(array.shape) == 1:
            return np.concatenate([array, np.zeros([inc])], 0)
        else:
            return np.concatenate([array, np.zeros([inc, array.shape[1]])], 0)

    def train_batch(self, sess, learning_rate, batch):
        feed_dict = self._get_feed_dict(batch)
        feed_dict[self.learning_rate] = learning_rate
        return sess.run([self.opt_op, self.merged_summary, self.global_step], feed_dict=feed_dict)

    def test_batch(self, sess, batch):
        params = self.params
        batch_size = params.batch_size

        actual_batch_size = len(batch[0])
        diff = batch_size - actual_batch_size
        if diff > 0:
            batch = [self._pad(each, diff) for each in batch]

        feed_dict = self._get_feed_dict(batch)
        correct_vec, total_loss, summary_str, global_step = \
            sess.run([self.correct_vec, self.total_loss, self.merged_summary, self.global_step], feed_dict=feed_dict)
        num_corrects = np.sum(correct_vec[:actual_batch_size])

        return num_corrects, total_loss, summary_str, global_step

    def _prepro_sent_batch(self, sent_batch):
        params = self.params
        N, M, J = len(sent_batch), params.memory_size, params.max_sent_size
        x_batch = np.zeros([N, M, J])
        x_mask_batch = np.zeros([N, M, J])
        m_mask_batch = np.zeros([N, M])
        for n, i, j in np.ndindex(x_batch.shape):
            if i < len(sent_batch[n]) and j < len(sent_batch[n][-i-1]):
                x_batch[n, i, j] = sent_batch[n][-i-1][j]
                x_mask_batch[n, i, j] = 1

        for n, i in np.ndindex(m_mask_batch.shape):
            if i < len(sent_batch[n]):
                m_mask_batch[n, i] = 1

        return x_batch, x_mask_batch, m_mask_batch

    def _prepro_ques_batch(self, ques_batch):
        params = self.params
        # FIXME : adhoc for now!
        N, J = len(ques_batch), params.max_sent_size
        q_batch = np.zeros([N, J])
        q_mask_batch = np.zeros([N, J])

        for n, j in np.ndindex(q_batch.shape):
            if j < len(ques_batch[n]):
                q_batch[n, j] = ques_batch[n][j]
                q_mask_batch[n, j] = 1

        return q_batch, q_mask_batch

    def _prepro_label_batch(self, label_batch):
        return np.array(label_batch)
