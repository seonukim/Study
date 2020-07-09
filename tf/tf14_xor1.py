import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(777)

## 데이터 정의
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

# x, y, w, b, h, cost, train
# sigmoid, predict, acc

x_col_num = x_data.shape[1]
y_col_num = y_data.shape[1]

## placeholder
x = tf.compat.v1.placeholder(tf.float32, shape = [None, x_col_num])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, y_col_num])
w = tf.compat.v1.Variable(tf.compat.v1.zeros([x_col_num, y_col_num]), name = 'weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([y_col_num]), name = 'bias')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
cost = -tf.compat.v1.reduce_mean(
    y * tf.compat.v1.log(hypothesis) + (1 - y) * tf.compat.v1.log(1 - hypothesis))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-6).minimize(cost)

predicted = tf.compat.v1.case(hypothesis > 0.5, tf.float32)     # tf.cast : if문과 비슷한 역할
acc = tf.compat.v1.reduce_mean(tf.compat.v1.cast(tf.compat.v1.equal(predicted, y), tf.float32))

init = tf.compat.v1.global_variables_initializer()
## 훈련
with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step i in range(2000 + 1):
        