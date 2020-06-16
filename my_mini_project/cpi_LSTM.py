import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU, LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
scaler = MinMaxScaler()
leaky = LeakyReLU(alpha = 0.3)
es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
modelpath = '/Users/seonwoo/Desktop/modelpath/LSTM/LSTM_{epoch:03d} - {loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath = modelpath, monitor = 'loss',
                     save_best_only = True, mode = 'auto',
                     save_weights_only = False, verbose = 1)


## 데이터
train = pd.read_csv('/Users/seonwoo/Downloads/'
                    '/cpi_train(1975.01 - 2002.09).csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
test = pd.read_csv('/Users/seonwoo/Downloads/'
                   '/cpi_test(2002.10 - 2020.05).csv',
                   index_col = 0, header = 0,
                   encoding = 'cp949')
print(train.shape)
print(test.shape)

## NumPy형 변환
train = train.values
test = test.values
print(type(train))      # <class 'numpy.ndarray'>
print(type(test))       # <class 'numpy.ndarray'>
print(train)


## 데이터 분할하기
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
    
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
'''
x, y = split_xy(train, 5, 1)
print(x[0, :], "\n", y[0])
print(x.shape)
print(y.shape)

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
model.add(LSTM(8, input_shape = (5, 13), activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

model.summary()


## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'rmsprop',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          verbose = 1, callbacks = [es, cp])


## 모델 평가
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])        # loss :  0.2592138775757381
print("mse : ", res[1])         # mse :  0.2592138648033142

## 예측
pred = model.predict(x_test)

for i in range(5):
    print('실제값 : ', y_test[i], '예측값 : ', pred[i])
'''
'''
실제값 :  [65.164] 예측값 :  [65.04013]
실제값 :  [65.055] 예측값 :  [65.1362]
실제값 :  [64.672] 예측값 :  [65.37116]
실제값 :  [64.452] 예측값 :  [65.382065]
실제값 :  [65.111] 예측값 :  [65.351524]
'''

x, y = split_xy(test, 5, 1)
print(x[0, :], "\n", y[0])
print(x.shape)
print(y.shape)

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

## 모델 로드
model = load_model('/Users/seonwoo/Desktop/modelpath/LSTM/LSTM_model.h5')
model.summary()

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'rmsprop',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          verbose = 1, callbacks = [es])

## 평가 및 예측
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])        # loss :  0.23455058889729635
print("mse : ", res[1])         # mse :  0.2345505952835083

## 예측
pred = model.predict(x_test)

for i in range(42):
    print('실제값 : ', y_test[i], '예측값 : ', pred[i])


'''
실제값 :  [102.64] 예측값 :  [100.51311]
실제값 :  [102.92] 예측값 :  [101.34612]
실제값 :  [102.85] 예측값 :  [102.060104]
실제값 :  [102.72] 예측값 :  [102.40213]
실제값 :  [102.83] 예측값 :  [102.98466]
실제값 :  [102.61] 예측값 :  [103.62777]
실제값 :  [102.78] 예측값 :  [103.81861]
실제값 :  [103.37] 예측값 :  [103.63492]
실제값 :  [103.49] 예측값 :  [104.06092]
실제값 :  [103.39] 예측값 :  [104.549675]
실제값 :  [102.62] 예측값 :  [104.906235]
실제값 :  [102.99] 예측값 :  [105.08491]
실제값 :  [103.42] 예측값 :  [105.25786]
실제값 :  [104.21] 예측값 :  [105.26211]
실제값 :  [104.1] 예측값 :  [105.129906]
실제값 :  [104.29] 예측값 :  [105.40243]
실제값 :  [104.34] 예측값 :  [105.88773]
실제값 :  [104.13] 예측값 :  [106.219826]
실제값 :  [103.93] 예측값 :  [106.41386]
실제값 :  [104.85] 예측값 :  [106.185104]
실제값 :  [105.65] 예측값 :  [106.1965]
실제값 :  [105.46] 예측값 :  [106.1134]
실제값 :  [104.71] 예측값 :  [106.13006]
실제값 :  [104.35] 예측값 :  [105.79035]
실제값 :  [104.24] 예측값 :  [106.2951]
실제값 :  [104.69] 예측값 :  [106.41172]
실제값 :  [104.49] 예측값 :  [105.81118]
실제값 :  [104.87] 예측값 :  [105.11559]
실제값 :  [105.05] 예측값 :  [104.647606]
실제값 :  [104.88] 예측값 :  [104.38388]
실제값 :  [104.56] 예측값 :  [104.264915]
실제값 :  [104.81] 예측값 :  [104.3361]
실제값 :  [105.2] 예측값 :  [104.65745]
실제값 :  [105.46] 예측값 :  [104.886475]
실제값 :  [104.87] 예측값 :  [104.72776]
실제값 :  [105.12] 예측값 :  [104.24771]
실제값 :  [105.79] 예측값 :  [104.68596]
실제값 :  [105.8] 예측값 :  [105.202065]
실제값 :  [105.54] 예측값 :  [105.20978]
실제값 :  [104.95] 예측값 :  [105.2318]
실제값 :  [104.71] 예측값 :  [105.61607]
실제값 :  [104.71] 예측값 :  [105.56021]        <- 2020.07월의 CPI 총 지수
'''
np.save('/Users/seonwoo/Desktop/modelpath/LSTM/x_train_cpi.npy', arr = x_train)
np.save('/Users/seonwoo/Desktop/modelpath/LSTM/x_test_cpi.npy', arr = x_test)
np.save('/Users/seonwoo/Desktop/modelpath/LSTM/y_train_cpi.npy', arr = y_train)
np.save('/Users/seonwoo/Desktop/modelpath/LSTM/y_test_cpi.npy', arr = y_test)
