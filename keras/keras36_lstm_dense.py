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
x = x.reshape(13, 3, 1)
x_predict = x_predict.reshape(1, 3, 1)

# print("x_reshape : ", x.shape)
# print("x_predict_reshape : ", x_predict.shape)

# 3. 모델 구성
# LSTM(return_sequences) _ Sequential 모델
'''
return_sequences 파라미터를 넣지 않고 에러가 나는 이유
LSTM은 3차원의 배열을 넣어줘야 하는데, (batch_size, timesteps, features)
우리가 입력한 차원은 2차원이다. ; Dense(row, column) -> 2차원
return_sequences는 이전 차원을 그대로 유지해주는 기능을 하는 parameter이다.
return_sequences의 값은 boolean, True of False
ValueError: Input 0 is incompatible with layer lstm_2: expectied ndim=3, found ndim=2
'''
model = Sequential()
model.add(LSTM(10, activation = 'relu', return_sequences = True,
               input_length = 3, input_dim = 1))
model.add(LSTM(10, return_sequences = False))
'''해당 레이어의 아웃풋 노드는 해당 아웃풋 레이어의 차원의 feature가 된다'''
'''증폭된 feature의 갯수는 아웃풋 노드의 갯수와 같다'''
'''아웃풋 노드의 개수는 그 다음 차원의 feature이다'''

''' 1) LSTM의 parameter 개수: 4 x (input_dim + bias(1) + output) x output
    2) GRU의 parameter 개수 : 3 x (input_dim + bias(1) + output) x output
    3) SimpleRNN의 parameter 개수 : 1 x (input_dim + bias(1) + output) x output
    ** 하나의 layer 당 bias는 1개(항상 존재함)
'''
# model.add(Flatten())  # 데이터를 강제로 2차원 변형 시킴.
# model.add(LSTM(24, return_sequences = True))
# model.add(LSTM(24))
# model.add(Dense(25))
# model.add(Dense(23))
# model.add(Dense(21))
# model.add(Dense(18))
# model.add(Dense(14))
# model.add(Dense(11))
# model.add(Dense(18))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

'''
# LSTM(return_sequences) _ 함수형 모델
input1 = Input(shape = (3, 1))
dense1 = LSTM(51, return_sequences = True)(input1)
dense2 = LSTM(48, return_sequences = True)(dense1)
dense3 = LSTM(49)(dense2)
dense4 = Dense(45)(dense3)
dense5 = Dense(45)(dense4)
dense6 = Dense(45)(dense5)
dense7 = Dense(45)(dense6)

output1 = Dense(31)(dense7)
output2 = Dense(16)(output1)
output3 = Dense(1)(output2)
output4 = Dense(1)(output3)
output5 = Dense(1)(output4)
output6 = Dense(1)(output5)

model = Model(inputs = input1,
              outputs = output6)
'''
'''
# 4. 실행
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x, y,
          epochs = 10000, batch_size = 10,
          callbacks = [early])

# 5. 예측
y_predict = model.predict(x_predict)

print(x_predict)
print(y_predict)
'''