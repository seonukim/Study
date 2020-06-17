import numpy as np
import warnings
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
'''
## Numpy 데이터 로드
x = np.load('./my_mini_project/npydata/cpi_train_x.npy')
y = np.load('./my_mini_project/npydata/cpi_train_y.npy')
print(x.shape)              # (208, 5, 13)
print(y.shape)              # (208, 1)

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.1, shuffle = False)
print(x_train.shape)        # (187, 5, 13)
print(x_test.shape)         # (21, 5, 13)
print(y_train.shape)        # (187, 1)
print(y_test.shape)         # (21, 1)

## Dense모델에 넣기 위해 reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (187, 65)
print(x_test.shape)         # (21, 65)

## Scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])

## 모델링
input1 = Input(shape = (65, ))
x = Dense(18, activation = 'relu')(input1)
x = Dropout(rate = 0.2)(x)
# x = Dense(8, activation = 'relu')(x)
# x = Dropout(rate = 0.2)(x)
# x = Dense(4, activation = 'relu')(x)
output = Dense(1, activation = 'relu')(x)

model = Model(inputs = input1, outputs = output)
model.summary()

model.save('./my_mini_project/Deep_Learning/Dense/Dense_function_model.h5')

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          verbose = 1, callbacks = [es])

## 모델 평가 및 예측
res = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", res[0])            # loss :  13.077937091570458
print("mse : ", res[1])             # mse :  13.077937126159668

x_pred = model.predict(x_test)
for i in range(5):
    print('실제값 : ', y_test[i], '예측값 : ', x_pred[i])
'''
'''
실제값 :  [68.103] 예측값 :  [67.771904]
실제값 :  [68.435] 예측값 :  [68.214745]
실제값 :  [69.035] 예측값 :  [68.87667]
실제값 :  [69.301] 예측값 :  [69.527306]
실제값 :  [69.234] 예측값 :  [70.345764]
'''

### 테스트 데이터를 이용하여 최종 예측
## Numpy 데이터 로드
x = np.load('./my_mini_project/npydata/cpi_test_x.npy')
y = np.load('./my_mini_project/npydata/cpi_test_y.npy')
print(x.shape)              # (208, 5, 13)
print(y.shape)              # (208, 1)

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.1, shuffle = False)
print(x_train.shape)        # (187, 5, 13)
print(x_test.shape)         # (21, 5, 13)
print(y_train.shape)        # (187, 1)
print(y_test.shape)         # (21, 1)

## Dense모델에 넣기 위해 reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (187, 65)
print(x_test.shape)         # (21, 65)

## Scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])

## 모델링
model = load_model('./my_mini_project/Deep_Learning/Dense/Dense_function_model.h5')
model.summary()

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'rmsprop',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          verbose = 1, callbacks = [es])

## 모델 평가 및 예측
res = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", res[0])            # loss :  836.1812765938895
print("mse : ", res[1])             # mse :  836.1812744140625

y_pred = model.predict(x_test)
for i in range(5):
    print('실제값 : ', y_test[i], '예측값 : ', y_pred[i])
'''
실제값 :  [105.46] 예측값 :  [119.759346]
실제값 :  [104.71] 예측값 :  [121.11568]
실제값 :  [104.35] 예측값 :  [120.512665]
실제값 :  [104.24] 예측값 :  [121.66885]
실제값 :  [104.69] 예측값 :  [123.37079]
'''