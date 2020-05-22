# keras35_lstm_sequences.py
# EarlyStopping을 적용하여 함수형 모델을 리모델링하시오

'''실습 : LSTM 레이어를 5개 이상 엮어서 Dense 결과를 이겨내시오'''

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
x = x.reshape(13, 3, 1)
x_predict = x_predict.reshape(1, 3, 1)


# print("x_reshape : ", x.shape)
# print("x_predict_reshape : ", x_predict.shape)

# 3. 모델 구성
# LSTM(return_sequences) _ Sequential 모델
'''
model = Sequential()
model.add(LSTM(300, activation = 'relu',
               input_length = 3, input_dim = 1,
               return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(30, return_sequences = True))
model.add(LSTM(40, return_sequences = True))
model.add(LSTM(20))
model.add(Dense(230))
model.add(Dense(20))
model.add(Dense(180))
model.add(Dense(14))
model.add(Dense(1))

model.summary()
'''


# LSTM(return_sequences) _ 함수형 모델
input1 = Input(shape = (3, 1))
dense1 = LSTM(60, return_sequences = True)(input1)
dense2 = LSTM(40, return_sequences = True)(dense1)
dense3 = LSTM(40, return_sequences = True)(dense2)
dense4 = LSTM(60, return_sequences = True)(dense3)
dense5 = LSTM(30)(dense4)
# dense6 = Dense(10)(dense5)
# dense7 = Dense(50)(dense6)
# dense8 = Dense(80)(dense7)
# dense9 = Dense(20)(dense8)
dense10 = Dense(10)(dense5)

output1 = Dense(30)(dense10)
output2 = Dense(60)(output1)
output3 = Dense(10)(output2)
output4 = Dense(40)(output3)
# output5 = Dense(80)(output4)
# output6 = Dense(90)(output5)
# output7 = Dense(10)(output6)
# output8 = Dense(50)(output7)
# output9 = Dense(90)(output8)
output10 = Dense(1)(output4)


model = Model(inputs = input1,
              outputs = output10)

model.summary()


# 4. 실행
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x, y,
          epochs = 10000, batch_size = 8,
          callbacks = [early])

# 5. 예측
y_predict = model.predict(x_predict)

print(x_predict)
print(y_predict)
