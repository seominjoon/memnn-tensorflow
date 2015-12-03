import os
import re
import logging


def read_data_set(file_path):
    with open(file_path, 'rb') as fh:
        s_re = re.compile("^(\\d+) ([\\w\\s\\?.]+)")
        q_re = re.compile("^(\\d+) ([\\w\\s\\?.]+)\t(\\w+)\t(\\d+)")
        paragraphs = []
        questions = []
        answers = []
        lines = fh.readlines()
        paragraph = []
        for line_num, line in enumerate(lines):
            sm = s_re.match(line)
            qm = q_re.match(line)
            if qm:
                id_, raw_question, answer, support = qm.groups()
                question = tokenize(raw_question)
                paragraphs.append(paragraph[:])
                questions.append(question)
                answers.append(answer)
            elif sm:
                id_, raw_sentence = sm.groups()
                sentence = tokenize(raw_sentence)
                if id_ == '1':
                    paragraph = []
                paragraph.append(sentence)
            else:
                logging.error("Invalid line encountered: %d" % (line_num + 1))
        print "Loaded %d examples from: %s" % (len(paragraphs), os.path.basename(file_path))
        return DataSet(paragraphs, questions, answers)


def get_vocabulary(data_set):
    paragraph_words = set(word for paragraph in data_set.paragraphs for sentence in paragraph for word in sentence)
    question_words = set(word for question in data_set.questions for word in question)
    answer_words = set(word for word in data_set.answers)
    return sorted(list(paragraph_words.union(question_words).union(answer_words)))


def tokenize(raw):
    return re.findall(r"[\w']+|[.,!?;]", raw)


class DataSet(object):
    def __init__(self, paragraphs, questions, answers):
        self.paragraphs = paragraphs
        self.questions = questions
        self.answers = answers
        assert len(paragraphs) == len(questions) and len(questions) == len(answers)
        self._num_examples = len(paragraphs)
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        i = self._index_in_epoch
        self._index_in_epoch += batch_size
        j = self._index_in_epoch
        return self.paragraphs[i:j], self.questions[i:j], self.answers[i:j]

    def has_next_batch(self, batch_size):
        return self._index_in_epoch + batch_size <= self._num_examples

if __name__ == "__main__":
    dir_path = "/Users/minjoon/workspace/memnn/memnn/data/tasks_1-20_v1-2/en/"
    vocab = set()
    for file_name in os.listdir(dir_path):
        if file_name.endswith("txt"):
            file_path = os.path.join(dir_path, file_name)
            ds = read_data_set(file_path)
            curr_vocab = get_vocabulary(ds)
            vocab = vocab.union(curr_vocab)
    print len(vocab)
    print vocab
