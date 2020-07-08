## 그래프로 확인해보기

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2. ,3.]
y = [1., 2., 3.]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w
cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y))

w_history = []
cost_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i * 0.1
        curr_cost = sess.run(cost, feed_dict = {w : curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

plt.plot(w_history, cost_history)
plt.show()