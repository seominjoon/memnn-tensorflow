import tensorflow as tf

import data
import memn2n_model

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 32, "Batch size during training [32]")
flags.DEFINE_integer("memory_size", 50, "Memory size [50]")
flags.DEFINE_integer("hidden_size", 20, "Embedding dimension [20]")
flags.DEFINE_integer("num_layer", 6, "Number of memory layers (hops) [6]")
flags.DEFINE_float("init_mean", 0.1, "Initial weight mean [0.1]")
flags.DEFINE_float("init_std", 0.05, "Initial weight std [0.05]")
flags.DEFINE_float("init_lr", 0.01, "Initial learning rate [0.01]")
flags.DEFINE_float("anneal_ratio", 0.5, "Annealing ratio [0.5]")
flags.DEFINE_integer("anneal_period", 25, "Number of epochs for every annealing [25]")
flags.DEFINE_float("max_grad_norm", 40, "Max gradient norm; above this number is clipped [40]")
flags.DEFINE_integer("num_epoch", 60, "Total number of epochs for training [60]")

flags.DEFINE_string("data_dir", 'data/tasks_1-20_v1-2/en/', "Data folder directory [data/tasks_1-20_v1-2/en]")
flags.DEFINE_string("data_prefix", "qa1_", "Prefix for file names to fetch in data_dir [qa1_]")
flags.DEFINE_string("data_suffix", "", "Suffix (before '_train.txt' or '_test.txt') for file names to fetch in data dir []")

FLAGS = flags.FLAGS



def main(_):
    # TODO : data read should get config file, and also pad zeros according to memory size, etc.
    train_ds, test_ds = data.read_babi(FLAGS)
    FLAGS.vocab_size = train_ds.vocab_size

    with tf.Session() as session:
        model = memn2n_model.MemN2NModel(FLAGS, session)
        model.train_data_set(train_ds, val_data_set=test_ds)

