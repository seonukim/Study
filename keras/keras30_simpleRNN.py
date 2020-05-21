from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.callbacks import EarlyStopping
import numpy as np
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

# 1. 데이터 구성
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6, 7])

# print(x.shape)      # res : (4, 3)
# print(y.shape)      # res : (4, )

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)        # res : (4, 3, 1)

print(x)


# 2. 모델 구성
model = Sequential()

model.add(SimpleRNN(10, activation = 'relu', input_shape = (3, 1)))
model.add(Dense(5))
model.add(Dense(1))

model.summary()
'''
model.summary()의 결과
Layer (type)                Output Shape    Param #
===================================================
simple_rnn_1 (SimpleRNN)    (None, 10)      (120)
---------------------------------------------------
dense_1 (Dense)             (None, 5)       (55)
---------------------------------------------------
dense_2 (Dense)             (None, 1)       (6)

 -> (input_dim * output)
  + (bias * output)
  + (output ^ 2)         * 1 -> SimpleRNN
  --------------------
  number of parameter (SimpleRNN)
'''

# 3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y, epochs = 100,
          batch_size = 1, callbacks = [es])

# 4. 실행
x_predict = np.array([5, 6, 7])
x_predict = x_predict.reshape(1, 3, 1)


# 5. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)