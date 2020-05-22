# keras35_lstm_sequences.py
# EarlyStopping을 적용하여 함수형 모델을 리모델링하시오

# 1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Flatten
from keras.callbacks import EarlyStopping
import numpy as np


# 1-1. EarlyStopping 객체 생성
early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)


# 2. 데이터 준비
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12],
              [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_predict = np.array([55, 65, 75])

print("x의 차원 : ", x.shape)
print("y의 차원 : ", y.shape)
print("x 예측값의 차원 : ", x_predict.shape)


# 2-1. 입력 데이터 reshape
# x = x.reshape(13, 3, 1)
x_predict = x_predict.reshape(1, 3)

# print("x_reshape : ", x.shape)
# print("x_predict_reshape : ", x_predict.shape)

# 3. 모델 구성 - Dense
model = Sequential()

model.add(Dense(30, activation = 'relu', input_shape = (3, )))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(1))

model.summary()


# 4. 실행
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x, y,
          epochs = 10000, batch_size = 10,
          callbacks = [early])

# 5. 예측
y_predict = model.predict(x_predict)

print(x_predict)
print(y_predict)
