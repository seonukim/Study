# mv : Multi Variables

import os
import tensorflow as tf
path = 'D:/Study/data/csv/'
os.chdir(path)
tf.compat.v1.set_random_seed(777)

x_data = [[1., 2.],
          [2., 3.],
          [3., 1.],
          [4., 3.],
          [5., 3.],
          [6., 2.]]
y_data = [[0.],
          [0.],
          [0.],
          [1.],
          [1.],
          [1.]]

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])
w = tf.compat.v1.Variable(tf.random.normal([2, 1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

# sigmoid = tf.compat.v1.div(1., 1. + tf.compat.v1.exp(tf.matmul(x, w)))
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)      # wx + b

# cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y_data))

## cost = sigmoid에 들어가는 손실함수 정의
cost = -tf.compat.v1.reduce_mean(
    y * tf.compat.v1.log(hypothesis) + (1 - y) * tf.compat.v1.log(1 - hypothesis))


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-6)
train = optimizer.minimize(cost)

predicted = tf.compat.v1.cast(hypothesis > 0.5, dtype = tf.float32)

# Accuracy 정의
accuracy = tf.compat.v1.reduce_mean(
    tf.compat.v1.cast(tf.compat.v1.equal(predicted, y), dtype = tf.float32))


fedict = {x: x_data, y: y_data}
init = tf.compat.v1.global_variables_initializer()      # 변수 초기화 - 선언하다
with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(10000 + 1):
        cost_val, _ = sess.run([cost, train], feed_dict = fedict)

        if step % 10 == 0:
            print(step, cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict = fedict)
    print(f'Hypothesis : {h}, \nCorrect : {c}, \nAccuracy : {a}')