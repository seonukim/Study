# 회귀
from sklearn.datasets import load_diabetes
import tensorflow as tf


## 데이터 불러오기
x_data, y_data = load_diabetes(return_X_y = True)
print(f'피쳐   : {x_data.shape} \n레이블 : {y_data.shape}')
'''
피쳐   : (442, 10)
레이블 : (442,)
'''

# 레이블 reshape
y_data = y_data.reshape(442, 1)
print(f'레이블 : {y_data.shape}')               # 레이블 : (442, 1)
print(y_data[:5])

## 텐서플로에 입력하기 위한 데이터 생성
# 1) x, y
# 2) w, b
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])
w = tf.compat.v1.Variable(tf.random.normal([10, 1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

## 선형 회귀를 위한 Hyper_parameter 생성
# 1) Activation = Linear
# 2) Loss function = mse
# 3) Optimizer = GradientDescent
hypothesis = tf.compat.v1.add(tf.compat.v1.matmul(x, w), b)
cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.5)

## 훈련 모델
train = optimizer.minimize(cost)

## 훈련
fedict = {x: x_data, y: y_data}
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(50000 + 1):
        cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict = fedict
        )

        if step % 10 == 0:
            print(f'{step}, 손실값 : {cost_val} \n{step}, 예측값 : \n{hy_val[:5]}\n')

'''
50000, 손실값 : 2862.923583984375
50000, 예측값 :                 실제값:
[[205.16641]                    [[151.]
 [ 68.91999]                     [ 75.]
 [175.83167]                     [141.]
 [165.46774]                     [206.]
 [128.24805]]                    [135.]]
 '''