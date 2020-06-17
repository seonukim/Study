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
x = Dense(16, activation = 'relu')(input1)
x = Dense(16, activation = 'relu')(x)
x = Dropout(rate = 0.2)(x)
x = Dense(8, activation = 'relu')(x)
x = Dense(8, activation = 'relu')(x)
x = Dropout(rate = 0.2)(x)
x = Dense(4, activation = 'relu')(x)
output = Dense(1, activation = 'relu')(x)

model = Model(inputs = input1, outputs = output)
model.summary()

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          verbose = 1, callbacks = [es])

## 모델 평가 및 예측
res = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", res[0])
print("mse : ", res[1])

