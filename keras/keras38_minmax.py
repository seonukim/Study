# keras38_minmax.py
# scikit_learn의 표준화, 정규화

# 1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Flatten
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1-1. 객체 생성
early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
scaler = MinMaxScaler()
standard = StandardScaler


# 2. 데이터 준비
# 전처리는 x만 한다.
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12],
              [2000, 3000, 4000], [3000, 4000, 5000], [4000, 5000, 6000],
              [100, 200, 300]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5000, 6000, 7000, 400])

x_predict = np.array([55, 65, 75])

scaler.fit(x)   # 실행하다
x = scaler.transform(x)
print(x)


print("x의 차원 : ", x.shape)                   # (14, 3)
print("y의 차원 : ", y.shape)                   # (14, )
print("x_predict 차원 : ", x_predict.shape)    # (3, )


# 2-1. 입력 데이터 reshape
x = x.reshape(14, 3, 1)
x_predict = x_predict.reshape(1, 3, 1)

# print("x_reshape : ", x.shape)
# print("x_predict_reshape : ", x_predict.shape)

'''
# 3. 모델 구성
# LSTM(return_sequences) _ Sequential 모델
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_length = 3, input_dim = 1,
               return_sequences = True))
model.add(LSTM(10, return_sequences = False))
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