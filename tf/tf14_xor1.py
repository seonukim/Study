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

## Model - Layer
w1 = tf.compat.v1.Variable(tf.compat.v1.zeros([x_col_num, 16]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), name = 'bias1')
layer1 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.compat.v1.zeros([16, 8]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name = 'bias2')
layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.compat.v1.zeros([8, 1]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name = 'bias3')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer2, w3) + b3)

cost = -tf.compat.v1.reduce_mean(
    y * tf.compat.v1.log(hypothesis) + (1 - y) * tf.compat.v1.log(1 - hypothesis))
    
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-6).minimize(cost)

predicted = tf.compat.v1.cast(hypothesis > 0.5, tf.float32)     # tf.cast : if문과 비슷한 역할
acc = tf.compat.v1.reduce_mean(tf.compat.v1.cast(tf.compat.v1.equal(predicted, y), tf.float32))


## 훈련
fedict = {x: x_data, y: y_data}
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(2000 + 1):
        cost_val, h_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict = fedict)

        if step % 20 == 0:
            print(step, cost_val)

    h, a, c = sess.run([hypothesis, predicted, acc], feed_dict = fedict)
    print(f'h : \n{h} \npred : \n{a} \nacc : {c}')

'''
h :
[[0.5]
 [0.5]
 [0.5]
 [0.5]]
pred :
[[0.]
 [0.]
 [0.]
 [0.]]
acc : 0.5
'''