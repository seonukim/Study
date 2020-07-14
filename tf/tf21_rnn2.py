import numpy as np
import tensorflow as tf

dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(dataset.shape) # (10,)

# RNN 모델을 짜시오
size = 5 

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i: (i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(dataset, size)
print(dataset)

x_data = dataset[:, 0:4]
print(x_data.shape) #(6,4)
print(x_data)
y_data = dataset[:, 4]
print(y_data.shape) #(6,)
print(y_data)

x_data = x_data.reshape(1, 6, 4)
y_data = y_data.reshape(6, 1)

sequence_length = 6
input_dim = 4
output = 4
batch_size = 1

X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
Y = tf.compat.v1.placeholder(tf.float32, (None, 1))

# 2. 모델 생성
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
print(hypothesis.shape) #(?, 6, 4)

w = tf.Variable(tf.random_normal([4, 1]), name = 'weights')
b = tf.Variable(tf.random_normal([1]), name = 'bias') 
hypothesis = tf.matmul(X, w) + b     


# 3.컴파일
# weights = tf.ones([batch_size, sequence_length])
cost =  tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)


# 3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1501):
        loss, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data}) #볼필요가 없으면 _만 사용한다.
        print(i,"loss :",loss)
    y_pred =sess.run(hypothesis, feed_dict = {X: x_data})
    print(y_pred)
