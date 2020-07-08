# tf06_1.py를 카피해서
# lr을 수정해서 연습
# 0.01 -> 0.1 / 0.001 / 1
# epoch가 2000번을 적게 만들어라

import tensorflow as tf

tf.compat.v1.set_random_seed(777)

# x_train, y_train 변수 생성
x_train = tf.compat.v1.placeholder(tf.float32, shape = [None])
y_train = tf.compat.v1.placeholder(tf.float32, shape = [None])

# feed_dict 변수 생성
fedict = {x_train: [1, 2, 3], y_train: [3, 5, 7]}

# Input(weight, bias, hypothesis) 생성
W = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

hypothesis = x_train * W + b

# 비용함수(손실함수) 정의
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# train 옵티마이저 정의(cost 최소화, GradientDescentOptimizer)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-1).minimize(cost)

## 훈련 및 예측
init = tf.compat.v1.global_variables_initializer()
pred_feed_1 = {x_train:[4]}
pred_feed_2 = {x_train:[5, 6]}
pred_feed_3 = {x_train:[6, 7, 8]}

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(1000 + 1):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = fedict)

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
    
    print(f'예측(1) : {sess.run(hypothesis, feed_dict = pred_feed_1)}')
    print(f'예측(2) : {sess.run(hypothesis, feed_dict = pred_feed_2)}')
    print(f'예측(3) : {sess.run(hypothesis, feed_dict = pred_feed_3)}')