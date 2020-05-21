# 함수형 모델로 리뉴얼하시오.

# 1. 모듈 임포트
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

# 2. 데이터 구성
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12],
              [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_predict = np.array([50, 60, 70])

# print("x.shape : ", x.shape)        # res : (13, 3)
# print("y.shape : ", y.shape)        # res : (13, )

x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape)
# print("x_predict : ", x_predict.shape)


# 3. 모델 구성
# 3-1. 인풋 레이어
input1 = Input(shape = (3, 1))
dense1 = Dense(5, activation = 'relu')(input1)
dense2 = Dense(3, activation = 'relu')(dense1)
dense3 = Dense(3, activation = 'relu')(dense2)
dense4 = Dense(2, activation = 'relu')(dense3)
dense5 = Dense(2, activation = 'relu')(dense4)

# 3-2. 아웃풋 레이어
output1 = Dense(2)(dense5)
output2 = Dense(1)(output1)

# 3-3. 모델링
model = Model(inputs = input1,
              outputs = output2)


# 4. 실행
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y,
          epochs = 10000, batch_size = 2,
          callbacks = [early])

x_predict = x_predict.reshape(1, 3, 1)


# 4. 평가 및 예측
# res = model.evaluate(x_test, y_test, batch_size = 1)
# print(res)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
