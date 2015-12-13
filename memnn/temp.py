import psycopg2
import os
import input_data
import time

conn = psycopg2.connect("dbname='memnn'")
cur = conn.cursor()

q_drop = """drop table memnn"""

q_create = """create table memnn(
did char(50),
pn int,
bn int,
sn int,
pos int,
word char(50));
"""
q_insert = """insert into memnn values ('%s', %d, %d, %d, %d, '%s')"""
q_select = """select t1.word, t2.word, count(*)
from memnn t1, memnn t2
where t1.did = t2.did and t1.pn = t2.pn and t1.bn = t2.bn and t1.sn = t2.sn and t1.pos + 1 = t2.pos
group by t1.word, t2.word
order by -count(*)"""

q_index = """create index on memnn (did, pn, bn, sn)"""

def populate():
    cur.execute(q_drop)
    cur.execute(q_create)

    dir_path = "/Users/minjoon/workspace/memnn/memnn/data/tasks_1-20_v1-2/en/"
    for file_name in os.listdir(dir_path):
        if file_name.endswith("txt"):
            file_path = os.path.join(dir_path, file_name)
            ds = input_data.read_data_set(file_path)
            for pn, paragraph in enumerate(ds.paragraphs):
                for sn, sentence in enumerate(paragraph):
                    for pos, word in enumerate(sentence):
                        cur.execute(q_insert % (file_name, 1, pn, sn, pos, word))
            break
    conn.commit()

def baseline():
    counts = {}
    dir_path = "/Users/minjoon/workspace/memnn/memnn/data/tasks_1-20_v1-2/en/"
    for file_name in os.listdir(dir_path):
        if file_name.endswith("txt"):
            file_path = os.path.join(dir_path, file_name)
            ds = input_data.read_data_set(file_path)
            for pn, paragraph in enumerate(ds.paragraphs):
                for sn, sentence in enumerate(paragraph):
                    for w1, w2 in zip(sentence[:-1], sentence[1:]):
                        if (w1, w2) in counts:
                            counts[(w1,w2)] += 1
                        else:
                            counts[(w1,w2)] = 1

    print max(counts.items(), key=lambda x: x[1])


def select():
    start_time = time.time()
    cur.execute(q_select)
    end_time = time.time()
    print end_time - start_time

    start_time = time.time()
    baseline()
    end_time = time.time()
    print end_time - start_time


# populate()
select()


