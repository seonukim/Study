## Tensorflow - RNN

import numpy as np
import tensorflow as tf

# data : hihello
idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype = np.str).reshape(-1, 1)
print(_data)
print(_data.shape)              # (7, 1)
print(type(_data))              # <class 'numpy.ndarray'>

# onehotencoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()
print("=" * 40)
print(_data)
print(type(_data))              # <class 'numpy.ndarray'>
print(_data.dtype)              # float64
'''
h i h e l l o 중,
1) h ~ l까지는 x
2) i ~ o까지는 y        로 잡는다.
'''

x_data = _data[:6, ]            # hihell
y_data = _data[1:, ]            #  ihello
print("=" * 40)
print("=" * 40)
print(f'x_data : \n{x_data}')
print("=" * 40)
print(f'y_data : \n{y_data}')
print("=" * 40)
'''
x_data :
[[0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]]
========================================
y_data :
[[0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
'''

y_data = np.argmax(y_data, axis = 1)
print("=" * 40)
print(f'y_data_argmax : \n{y_data}')
print(y_data.shape)
'''
y_data_argmax :
[2 1 0 3 3 4]
(6,)
'''

x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)
print(f'x_data.shape : {x_data.shape}')           # x_data.shape : (1, 6, 5)
print(f'y_data.shape : {y_data.shape}')           # y_data.shape : (1, 6)

sequence_length = 6
input_dim = 5
output = 5
batch_size = 1          # 전체 행

X = tf.compat.v1.placeholder(tf.float32, shape = (None, sequence_length, input_dim))
Y = tf.compat.v1.placeholder(tf.int64, shape = (None, sequence_length))
print(X)                # Tensor("Placeholder:0", shape=(?, 6, 5), dtype=float32)
print(Y)                # Tensor("Placeholder_1:0", shape=(?, 6), dtype=int64)


# 2. 모델 구성
# 케라스 형식
# model = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(output, input_shape = (6, 5))
# ])

# 텐서플로 1.14
# _lstm = tf.nn.rnn_cell.BasicLSTMCell(output)            # RNN은 두 번 연산함
_lstm = tf.keras.layers.LSTMCell(output) 
hypothesis, _states = tf.nn.dynamic_rnn(_lstm, X, dtype = tf.float32)        # 동적 RNN
                            # model.add(LSTM)        위 두 줄이 케라스의 model.add(LSTM)과 같다
print(hypothesis)           # Tensor("rnn/transpose_1:0", shape=(?, 6, 5), dtype=float32)

# 3. 컴파일
# loss 설정 : LSTM의 로스(hypothesis - y)
weights = tf.ones([batch_size, sequence_length])       # Y의 shape와 같음
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = Y, weights = weights)        # targets은 int32, 64형 들어가야 함

cost = tf.compat.v1.reduce_mean(sequence_loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.1).minimize(cost)

prediction = tf.math.argmax(hypothesis, axis = 2)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict = {X: x_data})
        print(f'\nEpoch : {i}, Prediction : {result}, true Y : {y_data}')

        result_str = [idx2char[c] for c in np.squeeze(result)]  # np.sqeeze() : 찾아보기
        print(f"\nPrediction : {''.join(result_str)}")       # ''.join() : 붙인다, 찾아보기