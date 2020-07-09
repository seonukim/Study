# 텐서플로우 - 전처리 preprocessing

import numpy as np
import tensorflow as tf

## Min_Max_Scaler 함수 정의
def min_max_scaler(dataset):
    numerator = dataset - np.min(dataset, 0)        # 데이커에서 최솟값을 뺌
    denominator = np.max(dataset, 0) - np.min(dataset, 0)
    return numerator / (denominator + 1e-7)         # 0으로 나눈다면 그것은 오류때문에 1e-7을 더해준다(고정은 아님)

## 데이터 정의
dataset = np.array( [ [828.659973, 833.450012, 908100, 828.349976, 831.659973],
                      [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                      [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                      [816, 820.958984, 1008100, 815.48999, 819.23999],
                      [819.359985, 823, 1188100, 818.469971, 818.97998],
                      [819, 823, 1198100, 816, 820.450012],
                      [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                      [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

## Scaler 적용하기
dataset = min_max_scaler(dataset = dataset)
print(dataset)

## 데이터 x, y로 나누기
x_data = dataset[:, :-1]
y_data = dataset[:, -1:]
print(x_data.shape)             # (8, 4)
print(y_data.shape)             # (8, 1)

x_col_num = x_data.shape[1]
y_col_num = y_data.shape[1]

## 텐서플로 하이퍼파라미터
x = tf.compat.v1.placeholder(tf.float32, shape = [None, x_col_num])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, y_col_num])
w = tf.compat.v1.Variable(tf.compat.v1.zeros([x_col_num, y_col_num]), name = 'weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([y_col_num]), name = 'bias')

hypothesis = tf.compat.v1.matmul(x, w) + b
cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5).minimize(cost)

## 훈련
fedict = {x: x_data, y: y_data}
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(2000 + 1):
        cost_val, h_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict = fedict)

        if step % 20 == 0:
            print(step, h_val, '\ncost : ', cost_val)
        print(y_data)
'''
cost :  0.30337995
[[1.        ]
 [0.83755791]
 [0.6606331 ]
 [0.43800918]
 [0.42624401]
 [0.49276137]
 [0.18597238]
 [0.        ]]
'''