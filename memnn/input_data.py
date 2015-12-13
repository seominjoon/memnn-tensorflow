import os
import re
import logging
import numpy as np


def read_data_set(file_path):
    with open(file_path, 'rb') as fh:
        s_re = re.compile("^(\\d+) ([\\w\\s.]+)")
        q_re = re.compile("^(\\d+) ([\\w\\s\\?]+)\t([\\w,]+)\t(\\d+)")
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


def tokenize(raw):
    return re.findall(r"[\w']+|[.,!?;]", raw)


class DataSet(object):
    def __init__(self, paragraphs, questions, answers):
        self.paragraphs = paragraphs
        self.questions = questions
        self.answers = answers
        assert len(paragraphs) == len(questions) and len(questions) == len(answers)
        self._num_examples = len(paragraphs)
        paragraph_words = set(word for paragraph in self.paragraphs for sentence in paragraph for word in sentence)
        question_words = set(word for question in self.questions for word in question)
        answer_words = set(word for word in self.answers)
        self.vocab = sorted(list(paragraph_words.union(question_words).union(answer_words)))

        self._indices = None
        self.shuffled_paragraphs = None
        self.shuffled_questions = None
        self.shuffled_answers = None
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self.shuffle()

    def shuffle(self):
        self._indices = range(self._num_examples)
        np.random.shuffle(self._indices)
        self.shuffled_paragraphs = [self.paragraphs[i] for i in self._indices]
        self.shuffled_questions = [self.questions[i] for i in self._indices]
        self.shuffled_answers = [self.questions[i] for i in self._indices]

    def next_batch(self, batch_size):
        i = self._index_in_epoch
        self._index_in_epoch += batch_size
        j = self._index_in_epoch
        if self._index_in_epoch > self._num_examples:
            self.shuffle()
            i = 0
            self._index_in_epoch = batch_size
            j = self._index_in_epoch
            self._epochs_completed += 1
        return self.shuffled_paragraphs[i:j], self.shuffled_questions[i:j], self.shuffled_answers[i:j]


if __name__ == "__main__":
    dir_path = "/Users/minjoon/workspace/memnn/memnn/data/tasks_1-20_v1-2/en/"
    vocab_set = set()
    for file_name in os.listdir(dir_path):
        if file_name.endswith("txt"):
            file_path = os.path.join(dir_path, file_name)
            ds = read_data_set(file_path)
            vocab_set = vocab_set.union(ds.vocab)
    print len(vocab_set)
    print vocab_set
