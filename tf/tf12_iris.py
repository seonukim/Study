import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

tf.compat.v1.set_random_seed(seed = 777)
sess = tf.compat.v1.Session()

## 데이터 불러오기
x_data, y_data = load_iris(return_X_y = True)
print(f'x_data : {x_data.shape} \ny_data : {y_data.shape}')
'''
x_data : (150, 4)
y_data : (150,)
'''

# 텐서플로의 원핫인코딩
# aaa = tf.one_hot(y, ???)
y_data = tf.compat.v1.one_hot(
    np.array(y_data, dtype = np.float32),
    depth = 3, on_value = 1, off_value = 0).eval(session = sess)
sess.close()            # Session 닫아주기


## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data,
    test_size = 0.2, shuffle = True)
print(f'x_train : {x_train.shape} \nx_test : {x_test.shape}')
'''
x_train : (120, 4)
x_test : (30, 4)
'''

x_col_num = x_train.shape[1]        # 4
y_col_num = y_train.shape[1]        # 3

## 텐서플로에 입력하기 위한 파라미터
x = tf.compat.v1.placeholder(tf.float32, shape = [None, x_col_num])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, y_col_num])
w = tf.compat.v1.Variable(tf.random.normal([x_col_num, y_col_num]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1, y_col_num]), name = 'bias')

hypothesis = tf.compat.v1.nn.softmax(
    tf.compat.v1.matmul(x, w) + b)

loss = tf.compat.v1.reduce_mean(
    tf.compat.v1.reduce_sum(y * tf.compat.v1.log(hypothesis), axis = 1))     # loss(categorical_crossentropy)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.000015).minimize(loss)
# optimizer = GradientDescentOptimizer ; 시간 있을 때 꼭 공부하기(경사하강법)

init = tf.compat.v1.global_variables_initializer()
fedict = {x: x_train, y: y_train}
## 훈련
with tf.compat.v1.Session() as sess:
    sess.run(init)

    for i in range(2000 + 1):
        _, _, cost_val, = sess.run([hypothesis, optimizer, loss], feed_dict = fedict)

        if i % 20 == 0:
            print(i, cost_val)

    pred = sess.run(hypothesis, feed_dict = {x: x_test})
    pred = sess.run(tf.math.argmax(pred, 1))          # tf.math.argmax(a, 1) 안의 값들 중, 가장 큰 값의 인덱스 반환

    y_test = sess.run(tf.math.argmax(y_test, 1))

    acc = tf.compat.v1.reduce_mean(
        tf.compat.v1.cast(tf.compat.v1.equal(pred, y_test), tf.float32))
    acc = sess.run(acc)

print(f'예측값 : {pred[:10]} \n실제값 : {y_test[:10]} \n정확도 : {acc}')
'''
예측값 : [1 1 1 1 1 1 1 1 1 1]
실제값 : [2 1 0 0 2 2 1 0 0 1]
정확도 : 0.2666666805744171
'''