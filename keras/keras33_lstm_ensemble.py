# 앙상블 모델로 리뉴얼하시오.
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
from keras.layers import concatenate
early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

# 1. 데이터 구성
x1 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
               [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
               [9, 10, 11], [10, 11, 12],
               [20, 30, 40], [30, 40, 50], [40, 50, 60]])
x2 = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60],
               [50, 60, 70], [60, 70, 80], [70, 80, 90], [80, 90, 100],
               [90, 100, 110], [100, 110, 120],
               [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x1_predict = np.array([55, 65, 75])
x2_predict = np.array([65, 75, 85])

print("x1.shape : ", x1.shape)        # res : (13, 3)
print("x2.shape : ", x2.shape)        # res : (13, 3)
print("y.shape : ", y.shape)        # res : (13, )


x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
#                    13           3       1

x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)
#                    13           3       1
# print(x1.shape)
# print(x2.shape)


# 2. 모델 구성
# 2-1. 인풋 레이어
input1 = Input(shape = (3, 1))
dense1 = LSTM(150)(input1)
dense1_1 = Dense(81)(dense1)
dense1_2 = Dense(88)(dense1_1)
dense1_3 = Dense(98)(dense1_2)
dense1_4 = Dense(48)(dense1_3)
dense1_5 = Dense(86)(dense1_4)

# 2-2 인풋 레이어
input2 = Input(shape = (3, 1))
dense2 = LSTM(194)(input2)
dense2_1 = Dense(488)(dense2)
dense2_2 = Dense(815)(dense2_1)
dense2_3 = Dense(128)(dense2_2)
dense2_4 = Dense(83)(dense2_3)
dense2_5 = Dense(88)(dense2_4)

# 2-3. 레이어 병합
merge1 = concatenate([dense1_5, dense2_5])
middle1 = Dense(51)(merge1)
middle2 = Dense(95)(middle1)
middle3 = Dense(58)(middle2)
middle4 = Dense(54)(middle3)

# 2-4. 아웃풋 레이어
output1 = Dense(50)(middle4)
output2 = Dense(148)(output1)
output3 = Dense(16)(output2)
output4 = Dense(168)(output3)
output5 = Dense(1)(output4)

# 2-5. 모델링
model = Model(inputs = [input1, input2],
              outputs = output5)

model.summary()



# 3. 실행
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1, x2], y,
          epochs = 1000, batch_size = 32)
        #   callbacks = [early])

x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)


# 4. 평가 및 예측
print(x1_predict)
print(x2_predict)

y_predict = model.predict([x1_predict, x2_predict])

print(y_predict)

'''
[[[55]
  [65]
  [75]]]
[[[65]
  [75]
  [85]]]
[[84.22178]]

[[84.29394]]

[[82.02529]]

[[83.22624]]

[[79.748474]]
'''