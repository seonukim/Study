import numpy as np
import tensorflow as tf
from keras.utils import np_utils

## 데이터 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)            # (60000, 28, 28)
print(x_test.shape)             # (10000, 28, 28)
print(y_train.shape)            # (60000,)
print(y_test.shape)             # (10000,)

## 데이터 전처리
# OneHotEncoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape, y_test.shape)          # (60000, 10) (10000, 10)

# 정규화
x_train = x_train.reshape(60000, 28 * 28).astype('float32') / 255
x_test = x_test.reshape(10000, 28 * 28).astype('float32') / 255
print(x_train.shape, x_test.shape)          # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape)          # (60000, 10) (10000, 10)

## 파라미터 정의
learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)        # 60000 / 100 = 600

x = tf.compat.v1.placeholder(tf.float32, [None, 784])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])
keep_prob = tf.compat.v1.placeholder(tf.float32)        # dropout

# w1 = tf.compat.v1.Variable(tf.random.normal([784, 512]), name = 'weight')     # 기존 Variable() 방식

## 모델 - 레이어 구성
w1 = tf.compat.v1.get_variable(
    'w1', shape = [784, 512],
    initializer = tf.contrib.layers.xavier_initializer())                       # get_variable() : Variable()보다 조금 더 좋은 녀석이다
b1 = tf.compat.v1.Variable(tf.random.normal([512]))
L1 = tf.compat.v1.nn.selu(
    tf.compat.v1.matmul(x, w1) + b1)
L1 = tf.compat.v1.nn.dropout(L1, keep_prob = keep_prob)                         # First hidden layer

w2 = tf.compat.v1.get_variable(
    'w2', shape = [512, 512],
    initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.compat.v1.Variable(tf.random.normal([512]))
L2 = tf.compat.v1.nn.selu(
    tf.compat.v1.matmul(L1, w2) + b2)
L2 = tf.compat.v1.nn.dropout(L2, keep_prob = keep_prob)                         # Second hidden layer

w3 = tf.compat.v1.get_variable(
    'w3', shape = [512, 512],
    initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.compat.v1.Variable(tf.random.normal([512]))
L3 = tf.compat.v1.nn.selu(
    tf.compat.v1.matmul(L2, w3) + b3)
L3 = tf.compat.v1.nn.dropout(L3, keep_prob = keep_prob)                         # Third hidden layer

w4 = tf.compat.v1.get_variable(
    'w4', shape = [512, 256],
    initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.compat.v1.Variable(tf.random.normal([256]))
L4 = tf.compat.v1.nn.selu(
    tf.compat.v1.matmul(L3, w4) + b4)
L4 = tf.compat.v1.nn.dropout(L4, rate = keep_prob)                              # Fourth hidden layer

w5 = tf.compat.v1.get_variable(
    'w5', shape = [256, 10],
    initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.compat.v1.Variable(tf.random.normal([10]))
hypothesis = tf.compat.v1.nn.softmax(
    tf.compat.v1.matmul(L4, w5) + b5)                                           # Output layer

cost = tf.compat.v1.reduce_mean(
    -tf.compat.v1.reduce_sum(y * tf.compat.v1.log(hypothesis), axis = 1))       # loss function

optimizer = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate = learning_rate).minimize(cost)                               # Optimization

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):            # 15
    avg_cost = 0                                # average cost : 평균 비용

    for i in range(total_batch):                # 600
        start = i * batch_size
        end = start + batch_size
#################### 이 부분 구현할 것 #######################
        batch_xs, batch_ys = x_train[start:end, :], y_train[start:end, :]
##############################################################
        feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 0.7}      # keep_prob : 0.7 -> 0.7만큼 남긴다
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch
    
    print(f'Epoch : {epoch+1:04d} \nCost : {avg_cost:.9f}')
print('훈련 끝')

prediction = tf.compat.v1.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(y, 1))
accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(prediction, tf.float32))
sess = tf.compat.v1.Session()
acc = sess.run(accuracy)

print(f'Accuracy : {acc}')     ### acc 출력할 것
sess.close() 