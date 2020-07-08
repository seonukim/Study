# 이진 분류
from sklearn.datasets import load_breast_cancer
import tensorflow as tf


## 데이터 불러오기
x_data, y_data = load_breast_cancer(return_X_y = True)
print(f'피쳐   : {x_data.shape} \n레이블 : {y_data.shape}')
'''
피쳐   : (569, 30)
레이블 : (569,)
'''

# 레이블 reshape
y_data = y_data.reshape(569, 1)
print(f'레이블 : {y_data.shape}')           # 레이블 : (569, 1)

## 텐서플로에 입력하기 위한 데이터 생성
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])
w = tf.compat.v1.Variable(tf.zeros([30, 1]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias')
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)
cost = -tf.compat.v1.reduce_mean(
    y * tf.compat.v1.log(hypothesis) + (1 - y) * tf.compat.v1.log(1 - hypothesis))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 5e-8)
train = optimizer.minimize(cost)

predicted = tf.compat.v1.cast(hypothesis > 0.5, dtype = tf.float32)

## Accuracy 정의
accuracy = tf.compat.v1.reduce_mean(
    tf.compat.v1.cast(tf.compat.v1.equal(predicted, y), dtype = tf.float32))


fedict = {x: x_data, y: y_data}
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(20000 + 1):
        cost_val, _ = sess.run([cost, train], feed_dict = fedict)

        if step % 10 == 0:
            print(step, cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict = fedict)
    print(f'\nHypothesis : \n{h[:5]}\n \nCorrect : \n{c[:5]}\n \nAccuracy : {a}')

'''
Hypothesis :
[[0.00428629]
 [0.08214608]
 [0.16400987]
 [0.4975429 ]
 [0.51463443]]

Correct :
[[0.]
 [0.]
 [0.]
 [0.]
 [1.]]

Accuracy : 0.9068541526794434
'''