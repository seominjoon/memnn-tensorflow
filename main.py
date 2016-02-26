from pprint import pprint
import os

import tensorflow as tf

import read_data
from models.base_model import BaseModel
from models.n2n_model import N2NModel

flags = tf.app.flags

# File directories
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("data_dir", 'data/tasks_1-20_v1-2/en/', "Data folder directory [data/tasks_1-20_v1-2/en]")
flags.DEFINE_string("save_dir", "save", "Save path [save]")

# Training parameters
flags.DEFINE_integer("batch_size", 32, "Batch size during training and testing [32]")
flags.DEFINE_integer("memory_size", 50, "Memory size [50]")
flags.DEFINE_integer("hidden_size", 20, "Embedding dimension [20]")
flags.DEFINE_integer("num_layers", 3, "Number of memory layers (hops) [3]")
flags.DEFINE_float("init_mean", 0, "Initial weight mean [0]")
flags.DEFINE_float("init_std", 0.1, "Initial weight std [0.1]")
flags.DEFINE_boolean("linear_start", True, "Start training with linear model? [True]")
flags.DEFINE_float("init_lr", 0.01, "Initial learning rate [0.01]")
flags.DEFINE_float("ls_init_lr", 0.005, "Initial learning rate for linear start [0.005]")
flags.DEFINE_integer("num_epochs", 100, "Total number of epochs for training [100]")
flags.DEFINE_integer("ls_num_epochs", 20, "Linear start duration [20]")
flags.DEFINE_float("anneal_ratio", 0.5, "Annealing ratio [0.5]")
flags.DEFINE_integer("anneal_period", 25, "Number of epochs for every annealing [25]")
flags.DEFINE_float("max_grad_norm", 40, "Max gradient norm; above this number is clipped [40]")
flags.DEFINE_boolean("position_encoding", True, "Position encoding enabled? 'True' or 'False' [True]")
flags.DEFINE_string("tying", 'adj', "Indicate tying method: 'adj' or 'rnn' [adj]")

# Training and testing options
flags.DEFINE_integer("task", 1, "Task number [1]")
flags.DEFINE_boolean("train", False, "Train? Test if False [False]")
flags.DEFINE_boolean("load", False, "Load from saved model? [False]")
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_boolean("gpu", False, 'Enable GPU? (Linux only) [False]')
flags.DEFINE_float("val_ratio", 0.1, "Validation data ratio to training data [0.1]")
flags.DEFINE_integer("eval_period", 20, "Val data eval period (for display purpose only) [20]")
flags.DEFINE_integer("save_period", 1, "Save period [1]")

# Debugging
flags.DEFINE_boolean("draft", False, "Draft? (quick build) [False]")

FLAGS = flags.FLAGS


def main(_):
    train_ds, test_ds = read_data.read_babi(FLAGS.batch_size, FLAGS.data_dir, FLAGS.task)
    train_ds, val_ds = read_data.split_val(train_ds, FLAGS.val_ratio)
    train_ds.name, val_ds.name, test_ds.name = 'train', 'val', 'test'
    FLAGS.vocab_size = test_ds.vocab_size
    FLAGS.max_sent_size, FLAGS.max_ques_size = read_data.get_max_sizes(train_ds, val_ds, test_ds)
    # FIXME : adhoc for now!
    FLAGS.max_sent_size = max(FLAGS.max_sent_size, FLAGS.max_ques_size)
    FLAGS.train_num_batches = train_ds.num_batches
    FLAGS.eval_num_batches = val_ds.num_batches
    FLAGS.test_num_batches = test_ds.num_batches
    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)

    if FLAGS.linear_start:
        FLAGS.num_epochs = FLAGS.ls_num_epochs
        FLAGS.init_lr = FLAGS.ls_init_lr

    if FLAGS.draft:
        FLAGS.num_layers = 1
        FLAGS.num_epochs = 1
        FLAGS.eval_period = 1
        FLAGS.ls_duration = 1
        FLAGS.train_num_batches = 1
        FLAGS.test_num_batches = 1
        FLAGS.save_period = 1

    pprint(FLAGS.__flags)
    print "training: %d, validation: %d, test: %d" % (train_ds.num_examples, val_ds.num_examples, test_ds.num_examples)

    graph = tf.Graph()
    model = N2NModel(graph, FLAGS)
    with tf.Session(graph=graph) as sess:
        sess.run(tf.initialize_all_variables())
        if FLAGS.train:
            writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph_def)
            if FLAGS.load:
                model.load(sess)
            model.train(sess, writer, train_ds, val_ds, num_batches=FLAGS.train_num_batches)
        else:
            model.load(sess)
            model.test(sess, test_ds, num_batches=FLAGS.test_num_batches)

if __name__ == "__main__":
    tf.app.run()
