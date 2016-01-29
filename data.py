import os
import re
import logging
from pprint import pprint
import numpy as np

class DataSet(object):
    def __init__(self, xs, qs, ys, max_sentence_size=None, vocab_size=None, vocab_map=None):
        assert len(xs) == len(ys), "X and Y sizes don't match."
        self.num_examples = len(xs)
        self.xs = xs
        self.qs = qs
        self.ys = ys
        self.max_sentence_size = max_sentence_size
        self.vocab_size = vocab_size
        self.vocab_map = vocab_map
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        assert batch_size <= self.num_examples, "batch size cannot be greater than data size."
        assert self.has_next(batch_size), "there is not batch left!"
        i, b = self._index_in_epoch, batch_size
        batch = self.xs[i:i + b], self.qs[i:i + b], self.ys[i:i + b]
        self._index_in_epoch += batch_size
        return batch

    def has_next(self, batch_size):
        return self._index_in_epoch + batch_size <= self.num_examples

    def rewind(self):
        self._index_in_epoch = 0
        self._epochs_completed += 1
        xqys = zip(self.xs, self.qs, self.ys)
        np.random.shuffle(xqys)
        self.xs, self.qs, self.ys = zip(*xqys)

# TODO : split data into val

def _tokenize(raw):
    return re.findall(r"[\w']+|[.,!?;]", raw)


_s_re = re.compile("^(\\d+) ([\\w\\s.]+)")
_q_re = re.compile("^(\\d+) ([\\w\\s\\?]+)\t([\\w,]+)\t(\\d+)")


def read_babi_files(file_paths):

    vocab_set = set()
    paragraphs = []
    questions = []
    answers = []

    for file_path in file_paths:
        with open(file_path, 'rb') as fh:
            lines = fh.readlines()
            paragraph = []
            for line_num, line in enumerate(lines):
                sm = _s_re.match(line)
                qm = _q_re.match(line)
                if qm:
                    id_, raw_question, answer, support = qm.groups()
                    question = _tokenize(raw_question)
                    paragraphs.append(paragraph[:])
                    questions.append(question)
                    answers.append(answer)
                    vocab_set |= set(question)
                    vocab_set.add(answer)
                elif sm:
                    id_, raw_sentence = sm.groups()
                    sentence = _tokenize(raw_sentence)
                    if id_ == '1':
                        paragraph = []
                    paragraph.append(sentence)
                    vocab_set |= set(sentence)
                else:
                    logging.error("Invalid line encountered: line %d in %s" % (line_num + 1, file_path))
            print "Loaded %d examples from: %s" % (len(paragraphs), os.path.basename(file_path))

    return vocab_set, paragraphs, questions, answers


def read_babi_split(*file_paths_list):
    vocab_set_list, paragraphs_list, questions_list, answers_list = zip(*[read_babi_files(file_paths) for file_paths in file_paths_list])
    max_sentence_size = max(max(len(sentence) for paragraphs in paragraphs_list for paragraph in paragraphs for sentence in paragraph),
                            max(len(question) for questions in questions_list for question in questions))

    vocab_set = vocab_set_list[0]
    vocab_map = dict((v, k+1) for k, v in enumerate(sorted(vocab_set)))
    vocab_map["<UNK>"] = 0

    def _get(vm, w):
        if w in vm:
            return vm[w]
        return 0

    xs_list = [[[[_get(vocab_map, word) for word in sentence] for sentence in paragraph] for paragraph in paragraphs] for paragraphs in paragraphs_list]
    qs_list = [[[_get(vocab_map, word) for word in question] for question in questions] for questions in questions_list]
    ys_list = [[_get(vocab_map, answer) for answer in answers] for answers in answers_list]

    data_sets = [DataSet(xs, qs, ys, max_sentence_size=max_sentence_size, vocab_size=len(vocab_map), vocab_map=vocab_map)
                 for xs, qs, ys in zip(xs_list, qs_list, ys_list)]
    return data_sets


def read_babi(dir_path, prefix="", suffix=""):
    train_file_paths = []
    test_file_paths = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if file_name.startswith(prefix) and file_name.endswith(suffix + "_train.txt"):
            train_file_paths.append(file_path)
        elif file_name.startswith(prefix) and file_name.endswith(suffix + "_test.txt"):
            test_file_paths.append(file_path)
    return read_babi_split(train_file_paths, test_file_paths)


def split_val(data_set, ratio):
    pass


if __name__ == "__main__":
    train, test = read_babi("data/tasks_1-20_v1-2/en", prefix="qa1_")
    # print train.vocab_size, train.max_m_len, train.max_s_len, train.max_q_len
    # print test.vocab_size, test.max_m_len, test.max_s_len, test.max_q_len
    x_batch, q_batch, y_batch = train.next_batch(10)
    print x_batch
    pprint(train.vocab_map)
