"""2020.05.18 수업내용 복습하기"""

# 1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np


# 2. 데이터 구성
x1 = np.array([range(1, 101), range(311, 411), range(100)])
y1 = np.array([range(711, 811), range(711, 811), range(100)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

# 2-1. 행렬 전치
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

# print(x1)
# print(x2)
# print(y1)
# print(y2)

# 2-2. 데이터 분할
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, test_size = 0.2, shuffle = False)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, test_size = 0.2, shuffle = False)
# print(x1_train)


# 3. 모델 구성
input1 = Input(shape = (3, ))
dense1 = Dense(5, activation = 'relu')(input1)
dense1_2 = Dense(4, activation = 'relu')(dense1)
dense1_3 = Dense(3)(dense1_2)

input2 = Input(shape = (3, ))
dense2 = Dense(5, activation = 'relu')(input2)
dense2_2 = Dense(4, activation = 'relu')(dense2)
dense2_3 = Dense(3)(dense2_2)

merge1 = concatenate([dense1_3, dense2_3])
middle1 = Dense(6)(merge1)
middle1 = Dense(8)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(8)(middle1)

output1 = Dense(4)(middle1)
output1_2 = Dense(5)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(4)(middle1)
output2_2 = Dense(5)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs = [input1, input2],
              outputs = [output1_3, output2_3])

# model.summary()


# 4. 훈련
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1_train, x2_train],
          [y1_train, y2_train], epochs = 1000, batch_size = 1,
          validation_split = 0.25, verbose = 3,
          callbacks = [es])

# 5. 평가 및 예측
res = model.evaluate([x1_test, x2_test],
                     [y1_test, y2_test], batch_size = 1)

# print(loss1)
# print(loss2)
# print(loss3)
# print(mse1)
# print(mse2)

y1_predict, y2_predict = model.predict([x1_test, x2_test])
# print(y1_predict)
# print(y2_predict)

# 6. RMSE 구하기
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse1 = RMSE(y1_test, y1_predict)
rmse2 = RMSE(y2_test, y2_predict)
# print("RMSE_1 : ", rmse1)
# print("-" * 40)
# print("RMSE_2 : ", rmse2)
print("-" * 40)

# 7. R2 구하기
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
# print("R2_1 : ", r2)
# print("-" * 40)
# print("R2_2 : ", r3)
print("RMSE : ", (rmse1 + rmse2) / 2)
print("-" * 40)
print("R2 : ", (r2_1 + r2_2) / 2)