import warnings ; warnings.filterwarnings('ignore')
import tensorflow as tf

x_train = [1, 2, 3]
y_train = [3, 5, 7]

W = tf.Variable(tf.random_normal(shape = [1], mean = 0,
                                 stddev = 1, name = 'weight'))
b = tf.Variable(tf.random_normal(shape = [1], mean = 0,
                                 stddev = 1, name = 'bias'))
'''
random_normal()
1) mean : 평균
2) stddev : 표준편차
'''

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # loss ; 손실함수
'''
reduce_mean() : 평균을 반환한다
square() : 제곱을 반환한다

hypothesis - y_train : 오차(예측값 - 실제값)
square(hypothesis - y_train) : 오차의 제곱

따라서, 위 코드는 mse(평균제곱오차)를 cost라는 변수로 정의한 코드
'''

## 훈련
train = tf.train.GradientDescentOptimizer(learning_rate = 1e-3).minimize(cost)
'''
train : 훈련함
GradientDescentOptimizer : 경사하강 최적화
minimize() : 인자로 들어갈 값을 최소화

따라서, 위 코드는 cost(비용함수)를 최소화하는 지점을 찾는데,
그 최소비용 찾는 최적화 방법을 경사하강법으로 하여
훈련하겠다; 라는 의미
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000 + 1):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
'''
with문을 사용하면 with 블록을 벗어나는 순간 자동으로 close함
'''