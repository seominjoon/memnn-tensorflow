import tensorflow as tf

class Model(object):
    def __init__(self, params):
        self.params = params
        self.variables = self._init_variables()
        self.train_graph = self._build_graph('train')
        self.test_graph = self._build_graph('test')

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
                        A = tf.get_variable('A', [V, d])
                        C = tf.get_variable('C', [V, d])
                    else:
                        if params.tying == 'adj':
                            A = Cs[-1]
                            C = tf.get_variable('C', [V, d])
                        elif params.tying == 'rnn':
                            A = As[-1]
                            C = Cs[-1]
                        else:
                            raise Exception('Unknown tying method: %s' % params.tying)
                    As.append(A)
                    Cs.append(C)

            if params.tying == 'adj':
                B = tf.no_op(As[0], name='B')
                W = tf.transpose(Cs[-1], name='W')
            elif params.tying == 'rnn':
                B = tf.get_variable('B', [V, d])
                W = tf.get_variable('W', [d, V])
            else:
                raise Exception('Unknown tying method: %s' % params.tying)

            variables.As, variables.B, variables.Cs, variables.W = As, B, Cs, W

        return variables

    def _build_graph(self, mode):
        params = self.params
        variables = self.variables
        M, J, d = params.memory_size, params.max_sentence_size, params.hidden_size
        if mode == 'train':
            N = params.train_batch_size
        elif mode == 'test':
            N = params.test_data_size
        else:
            raise Exception("Invalid graph mode: %s" % mode)

        As, B, Cs, W = variables.As, variables.B, variables.Cs, variables.W

        class Tensors(object):
            pass
        tensors = Tensors()

        # initialize tensors
        with tf.variable_scope(mode):
            # placeholders
            with tf.name_scope('ph'):
                x_batch = tf.placeholder('int32', shape=[N, M, J])
                q_batch = tf.placeholder('int32', shape=[N, J])
                y_batch = tf.placeholder('int32', shape=[N])

            Bq_batch = tf.nn.embedding_lookup(B, q_batch)  # [N, J, d]
            first_u_batch = tf.reduce_sum(Bq_batch, 1, name='first_u')  # [N, d]

            u_batch_list = []
            o_batch_list = []
            for layer_index in xrange(params.num_layers):
                with tf.name_scope('l%d' % layer_index):
                    if layer_index == 0:
                        u_batch = tf.no_op(first_u_batch, name='u')
                    else:
                        u_batch = tf.add(u_batch_list[-1], o_batch_list[-1], name='u')

                    Ax_batch = tf.nn.embedding_lookup(As[layer_index], x_batch)
                    m_batch = tf.reduce_sum(Ax_batch, 1, name='m')  # [N M d]
                    Cx_batch = tf.nn.embedding_lookup(Cs[layer_index], x_batch)
                    c_batch = tf.reduce_sum(Cx_batch, 1, name='c')

                    u_batch_x = tf.expand_dims(u_batch, -1)  # [N, d, 1]
                    p_batch_x = tf.nn.softmax(tf.batch_matmul(m_batch, u_batch_x))  # [N, M, 1]
                    p_batch = tf.squeeze(p_batch_x, [2], name='p')  # [N, M], just for reference

                    o_batch = tf.reduce_sum(c_batch * p_batch_x, 1)  # [N, d]

                u_batch_list.append(u_batch)
                o_batch_list.append(o_batch)

            last_u_batch = tf.add(u_batch_list[-1], o_batch_list[-1], name='last_u')
            ap_batch = tf.nn.softmax(tf.matmul(last_u_batch, W), name='ap')  # [N d] X [d V] = [N V]


        return tensors

    def train(self, x_batch, q_batch, y_batch):
        params = self.params

    def test(self, data_set):
        params = self.params