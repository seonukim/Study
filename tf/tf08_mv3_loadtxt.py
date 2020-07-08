import os
import numpy as np
import tensorflow as tf
path = 'D:/Study/data/csv/'
os.chdir(path)
tf.compat.v1.set_random_seed(777)

dataset = np.loadtxt(path + 'data-01-test-score.csv',
                     delimiter = ',', dtype = np.float32)

x_data = dataset[:, :-1]
y_data = dataset[:, -1:]
print(x_data.shape, y_data.shape)           # (25, 3) (25, 1)


x = tf.compat.v1.placeholder(tf.float32, shape = [None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])
w = tf.compat.v1.Variable(tf.random.normal([3, 1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

hypothesis = tf.compat.v1.add(tf.compat.v1.matmul(x, w), b)      # wx + b ; Linear Activation

cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y_data))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)


fedict = {x: x_data, y: y_data}
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(6000 + 1):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train], feed_dict = fedict
            )

        if step % 1 == 0:
            print(f'{step}, 손실값 : {cost_val} \n{step}, 예측값 : \n{hy_val}\n')