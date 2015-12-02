import input_data
import tensorflow as tf
import math

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.truncated_normal([784, 10], stddev=1.0/math.sqrt(784)))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

sess.run(tf.initialize_all_variables())

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in xrange(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

