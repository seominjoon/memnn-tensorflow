import os
import re
import logging


class DataSet(object):
    def __init__(self, ms, qs, ys, vocab_size=None):
        assert len(ms) == len(ys), "X and Y sizes don't match."
        self._num_examples = len(ms)
        self.ms = ms
        self.qs = qs
        self.ys = ys
        self.vocab_size = vocab_size
        self.max_m_len = max(len(m) for m in ms)
        self.max_s_len = max(len(mi) for m in ms for mi in m)
        self.max_q_len = max(len(q) for q in qs)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples, "batch size cannot be greater than data size."
        if self._index_in_epoch + batch_size > self._num_examples:
            self._index_in_epoch = 0
            self._epochs_completed += 1
        i, b = self._index_in_epoch, batch_size
        batch = self.ms[i:i + b], self.qs[i:i + b], self.ys[i:i + b]
        self._index_in_epoch += batch_size
        return batch


def _tokenize(raw):
    return re.findall(r"[\w']+|[.,!?;]", raw)


def read_babi_files(file_paths):
    s_re = re.compile("^(\\d+) ([\\w\\s.]+)")
    q_re = re.compile("^(\\d+) ([\\w\\s\\?]+)\t([\\w,]+)\t(\\d+)")

    vocab_set = set()
    paragraphs = []
    questions = []
    answers = []

    for file_path in file_paths:
        with open(file_path, 'rb') as fh:
            lines = fh.readlines()
            paragraph = []
            for line_num, line in enumerate(lines):
                sm = s_re.match(line)
                qm = q_re.match(line)
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

    vocab_map = dict((v, k) for k, v in enumerate(sorted(vocab_set)))
    xs = [[[vocab_map[word] for word in sentence] for sentence in paragraph] for paragraph in paragraphs]
    qs = [[vocab_map[word] for word in question] for question in questions]
    ys = [vocab_map[answer] for answer in answers]
    data_set = DataSet(xs, qs, ys, vocab_size=len(vocab_map))
    data_set.vocab_map = vocab_map

    return data_set


def read_babi_all(dir_path, prefix="", suffix=""):
    train_file_paths = []
    test_file_paths = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if file_name.startswith(prefix) and file_name.endswith(suffix + "_train.txt"):
            train_file_paths.append(file_path)
        elif file_name.startswith(prefix) and file_name.endswith(suffix + "_test.txt"):
            test_file_paths.append(file_path)
    train_data_set = read_babi_files(train_file_paths)
    test_data_set = read_babi_files(test_file_paths)
    return train_data_set, test_data_set


def read_babi(size='small', prefix="", suffix=""):
    if size == 'small':
        dir_path = "data/tasks_1-20_v1-2/en/"
    elif size == 'large':
        dir_path = "data/tasks_1-20_v1-2/en-10k/"
    else:
        raise Exception("Invalid size. Choose 'small' or 'large' (10k).")
    return read_babi_all(dir_path, prefix=prefix, suffix=suffix)


if __name__ == "__main__":
    train, test = read_babi(prefix="")
    print train.vocab_size, train.max_m_len, train.max_s_len, train.max_q_len
    print test.vocab_size, test.max_m_len, test.max_s_len, test.max_q_len
