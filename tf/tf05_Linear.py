import warnings ; warnings.filterwarnings('ignore')
import tensorflow as tf
tf.set_random_seed(777)

x_train = [1, 2, 3]
y_train = [3, 5, 7]

W = tf.Variable(tf.random_normal([1]), name = 'weight')     # 랜덤 정규분포
b = tf.Variable(tf.random_normal([1], name = 'bias'))       # [] 안의 숫자는 shape를 의미한다

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())     # 초기화 ; tf는 초기화를 시켜주어야 한다
# print(sess.run(W))

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))      # 비용함수(loss) 정의 ; 이건 mse이다

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
# GradientDescentOptimizer = 경사하강 최적화    minimize(cost) ; cost를 최소화하는 지점을 찾아라

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())         # with문 안에 나오는 모든 변수들을 초기화

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])       # keras의 compile ; optimizer 부분이다

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

'''
Session()을 열어주면, 작업이 끝난 후 무조건 Session을 close해야 한다
'''