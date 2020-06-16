import numpy as np
import warnings
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
warnings.filterwarnings('ignore')
es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# sclaer = RobustScaler()

'''
## Numpy 데이터 로드
x = np.load('./my_mini_project/npydata/cpi_train_x.npy')
y = np.load('./my_mini_project/npydata/cpi_train_y.npy')
print(x.shape)      # (208, 5, 13)
print(y.shape)      # (208, 1)

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.1, shuffle = False)
print(x_train.shape)        # (187, 5, 13)
print(x_test.shape)         # (21, 5, 13)
print(y_train.shape)        # (187, 1)
print(y_test.shape)         # (21, 1)

## reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (187, 65)
print(x_test.shape)         # (21, 65)

## Scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])

## reshape
x_train = x_train.reshape(187, 5, 13)
x_test = x_test.reshape(21, 5, 13)
print(x_train.shape)        # (187, 5, 13)
print(x_test.shape)         # (21, 5, 13)

## 모델링
model = Sequential()
model.add(GRU(8, input_shape = (5, 13),
          activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(1, activation = 'relu'))
model.summary()

model.save('./my_mini_project/GRU/GRU_model.h5')

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'rmsprop',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          callbacks = [es], verbose = 1)

## 모델 평가 및 예측
res = model.evaluate(x_test, y_test)
print('loss : ', res[0])        # loss :  3.9167377948760986
print('mse : ', res[1])         # mse :  3.9167377948760986

x_pred = model.predict(x_test)
for i in range(5):
    print('실제값 : ', y_test[i], '예측값 : ', x_pred[i])
'''
'''
실제값 :  [68.103] 예측값 :  [64.30937]
실제값 :  [68.435] 예측값 :  [65.11477]
실제값 :  [69.035] 예측값 :  [64.99606]
실제값 :  [69.301] 예측값 :  [65.40334]
실제값 :  [69.234] 예측값 :  [67.36651]
'''

### test 데이터로 최종 예측
## Numpy 데이터 로드
x = np.load('./my_mini_project/npydata/cpi_test_x.npy')
y = np.load('./my_mini_project/npydata/cpi_test_y.npy')
print(x.shape)      # (208, 5, 13)
print(y.shape)      # (208, 1)

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.1, shuffle = False)
print(x_train.shape)        # (187, 5, 13)
print(x_test.shape)         # (21, 5, 13)
print(y_train.shape)        # (187, 1)
print(y_test.shape)         # (21, 1)

## reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (187, 65)
print(x_test.shape)         # (21, 65)

## Scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])

## reshape
x_train = x_train.reshape(187, 5, 13)
x_test = x_test.reshape(21, 5, 13)
print(x_train.shape)        # (187, 5, 13)
print(x_test.shape)         # (21, 5, 13)

## 모델링
model = load_model('./my_mini_project/GRU/GRU_model.h5')
model.summary()

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'rmsprop',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          callbacks = [es], verbose = 1)

## 모델 평가 및 예측
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])        # loss :  176.84942626953125
print("mse : ", res[1])         # loss :  176.84942626953125

y_pred = model.predict(x_test)
for i in range(21):
    print("실제값 : ", y_test[i], '예측값 : ', y_pred[i])
'''
실제값 :  [105.46] 예측값 :  [108.84617]
실제값 :  [104.71] 예측값 :  [108.38826]
실제값 :  [104.35] 예측값 :  [109.04182]
실제값 :  [104.24] 예측값 :  [112.19602]
실제값 :  [104.69] 예측값 :  [113.93867]
실제값 :  [104.49] 예측값 :  [112.92871]
실제값 :  [104.87] 예측값 :  [112.70444]
실제값 :  [105.05] 예측값 :  [111.8336]
실제값 :  [104.88] 예측값 :  [113.77715]
실제값 :  [104.56] 예측값 :  [117.034225]
실제값 :  [104.81] 예측값 :  [117.50689]
실제값 :  [105.2] 예측값 :  [118.75825]
실제값 :  [105.46] 예측값 :  [119.11128]
실제값 :  [104.87] 예측값 :  [117.652664]
실제값 :  [105.12] 예측값 :  [117.96128]
실제값 :  [105.79] 예측값 :  [118.61997]
실제값 :  [105.8] 예측값 :  [119.24046]
실제값 :  [105.54] 예측값 :  [122.57953]
실제값 :  [104.95] 예측값 :  [124.4044]
실제값 :  [104.71] 예측값 :  [128.18826]
실제값 :  [104.71] 예측값 :  [130.77966]
'''