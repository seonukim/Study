import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU, LSTM
from keras.layers import Conv1D, Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
scaler = MinMaxScaler()
leaky = LeakyReLU(alpha = 0.3)
es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
# modelpath = './my_mini_project/LSTM/LSTM_{epoch:03d} - {loss:.4f}.hdf5'
# cp = ModelCheckpoint(filepath = modelpath, monitor = 'loss',
#                      save_best_only = True, mode = 'auto',
#                      save_weights_only = False, verbose = 1)

'''
## 데이터
x = np.load('./my_mini_project/npydata/cpi_train_x.npy')
y = np.load('./my_mini_project/npydata/cpi_train_y.npy')

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = False)
print(x_train.shape)        # (166, 5, 13)
print(x_test.shape)         # (42, 5, 13)
print(y_train.shape)        # (166, 1)
print(y_test.shape)         # (42, 1)

## 데이터 리쉐이프
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (166, 65)
print(x_test.shape)         # (42, 65)

## Scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

## 3차원으로 reshape
x_train = x_train.reshape(166, 5, 13)
x_test = x_test.reshape(42, 5, 13)


## 모델링
model = Sequential()
model.add(Conv1D(filters = 16, kernel_size = (3, ),
                 padding = 'same', input_shape = (5, 13),
                 activation = 'relu'))
model.add(Flatten())
model.add(Dense(1, activation = 'relu'))

model.summary()

model.save('./my_mini_project/Deep_Learning/CNN/cpi_CNN_model.h5')


## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          verbose = 1, callbacks = [es])


## 모델 평가
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])        # loss :  0.7944057214827764
print("mse : ", res[1])         # loss :  0.7944057214827764

## 예측
pred = model.predict(x_test)

for i in range(5):
    print('실제값 : ', y_test[i], '예측값 : ', pred[i])
'''
'''
실제값 :  [65.164] 예측값 :  [65.09349]
실제값 :  [65.055] 예측값 :  [65.32321]
실제값 :  [64.672] 예측값 :  [65.46494]
실제값 :  [64.452] 예측값 :  [65.42189]
실제값 :  [65.111] 예측값 :  [65.35302]
'''


## 데이터
x = np.load('./my_mini_project/npydata/cpi_test_x.npy')
y = np.load('./my_mini_project/npydata/cpi_test_y.npy')

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = False)
print(x_train.shape)        # (166, 5, 13)
print(x_test.shape)         # (42, 5, 13)
print(y_train.shape)        # (166, 1)
print(y_test.shape)         # (42, 1)

## 데이터 리쉐이프
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (166, 65)
print(x_test.shape)         # (42, 65)

## Scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

## 3차원으로 reshape
x_train = x_train.reshape(166, 5, 13)
x_test = x_test.reshape(42, 5, 13)

## 모델링
model = load_model('./my_mini_project/Deep_Learning/CNN/cpi_CNN_model.h5')
model.summary()

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          verbose = 1, callbacks = [es])

## 모델 평가
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])        # loss :  10.401627038206373
print("mse : ", res[1])         # mse :  10.401627540588379

## 예측
pred = model.predict(x_test)

for i in range(5):
    print('실제값 : ', y_test[i], '예측값 : ', pred[i])
'''
실제값 :  [102.64] 예측값 :  [103.11145]
실제값 :  [102.92] 예측값 :  [103.849556]
실제값 :  [102.85] 예측값 :  [104.56963]
실제값 :  [102.72] 예측값 :  [104.81483]
실제값 :  [102.83] 예측값 :  [105.16775]
'''