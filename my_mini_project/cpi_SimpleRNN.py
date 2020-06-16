import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU, SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)


## 데이터
train = pd.read_csv('C:/Users/bitcamp/Downloads/'
                    '/cpi_train(1975.01 - 2002.09).csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
test = pd.read_csv('C:/Users/bitcamp/Downloads/'
                   '/cpi_test(2002.10 - 2020.05).csv',
                   index_col = 0, header = 0,
                   encoding = 'cp949')
print(train.shape)      # (213, 13)
print(test.shape)       # (213, 13)

## Numpy형 변환
train = train.values
test = test.values
print(train.shape)      # (213, 13)
print(test.shape)       # (213, 13)

## 데이터 분할하기
def split_xy(data, time, y_column):
    x, y = list(), list()
    for i in range(len(data)):
        x_end_number = i + time
        y_end_number = x_end_number + y_column

        if y_end_number > len(data):
            break
        tmp_x = data[i:x_end_number, :]
        tmp_y = data[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
'''
x, y = split_xy(train, 5, 1)
print(x.shape)      # (208, 5, 13)
print(y.shape)      # (208, 1)

# Numpy 데이터 저장
np.save('./my_mini_project/npydata/cpi_train_x.npy', arr = x)
np.save('./my_mini_project/npydata/cpi_train_y.npy', arr = y)

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = False)
print(x_train.shape)        # (166, 5, 13)
print(x_test.shape)         # (42, 5, 13)
print(y_train.shape)        # (166, 1)
print(y_test.shape)         # (42, 1)

## Scaling을 위한 reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (166, 65)
print(x_test.shape)         # (42, 65)

## Scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])

## 다시 reshape
x_train = x_train.reshape(166, 5, 13)
x_test = x_test.reshape(42, 5, 13)
print(x_train.shape)        # (166, 5, 13)
print(x_test.shape)         # (42, 5, 13)

## 모델링
model = Sequential()
model.add(SimpleRNN(10, input_shape = (5, 13),
                    activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(1, activation = 'relu'))

model.summary()

model.save('./my_mini_project/SimpleRNN/SimpleRNN_model.h5')

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          callbacks = [es], verbose = 1)

## 모델 평가
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])        # loss :  2.749787035442534
print("mse : ", res[1])         # mse :  2.7497870922088623

x_pred = model.predict(x_test)
for i in range(5):
    print('실제값 : ', y_test[i], '예측값 : ', x_pred[i])
'''
'''
실제값 :  [65.164] 예측값 :  [65.7443]
실제값 :  [65.055] 예측값 :  [65.7999]
실제값 :  [64.672] 예측값 :  [65.429504]
실제값 :  [64.452] 예측값 :  [66.3179]
실제값 :  [65.111] 예측값 :  [65.95752]
'''

### 테스트 데이터로 최종 예측하기
## 데이터 분할
x, y = split_xy(test, 5, 1)
print(x.shape)      # (208, 5, 13)
print(y.shape)      # (208, 1)

## NumPy 저장
np.save('./my_mini_project/npydata/cpi_test_x.npy', arr = x)
np.save('./my_mini_project/npydata/cpi_test_y.npy', arr = y)

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = False)
print(x_train.shape)        # (166, 5, 13)
print(x_test.shape)         # (42, 5, 13)
print(y_train.shape)        # (166, 1)
print(y_test.shape)         # (42, 1)

## Scaling을 위한 reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (166, 65)
print(x_test.shape)         # (42, 65)

## Scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])

## 다시 reshape
x_train = x_train.reshape(166, 5, 13)
x_test = x_test.reshape(42, 5, 13)
print(x_train.shape)        # (166, 5, 13)
print(x_test.shape)         # (42, 5, 13)

## 모델링
model = load_model('./my_mini_project/SimpleRNN/SimpleRNN_model.h5')

model.summary()

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(x_train, y_train, epochs = 200,
          batch_size = 1, verbose = 1)

## 모델 평가 및 예측
res = model.evaluate(x_test, y_test)
print('loss : ', res[0])        # loss :  391.1343703497024
print('mse : ', res[1])         # mse :  391.1343994140625

y_pred = model.predict(x_test)
for i in range(42):
    print('실제값 : ', y_test[i], '예측값 : ', y_pred[i])
'''
실제값 :  [102.64] 예측값 :  [108.67034]
실제값 :  [102.92] 예측값 :  [109.96777]
실제값 :  [102.85] 예측값 :  [110.69697]
실제값 :  [102.72] 예측값 :  [111.24311]
실제값 :  [102.83] 예측값 :  [113.25305]
실제값 :  [102.61] 예측값 :  [116.61834]
실제값 :  [102.78] 예측값 :  [117.36192]
실제값 :  [103.37] 예측값 :  [117.31978]
실제값 :  [103.49] 예측값 :  [116.75168]
실제값 :  [103.39] 예측값 :  [116.815544]
실제값 :  [102.62] 예측값 :  [116.8555]
실제값 :  [102.99] 예측값 :  [117.670746]
실제값 :  [103.42] 예측값 :  [118.33843]
실제값 :  [104.21] 예측값 :  [118.60556]
실제값 :  [104.1] 예측값 :  [118.5932]
실제값 :  [104.29] 예측값 :  [118.01661]
실제값 :  [104.34] 예측값 :  [119.662]
실제값 :  [104.13] 예측값 :  [121.37894]
실제값 :  [103.93] 예측값 :  [122.91931]
실제값 :  [104.85] 예측값 :  [123.39589]
실제값 :  [105.65] 예측값 :  [124.08857]
실제값 :  [105.46] 예측값 :  [124.4527]
실제값 :  [104.71] 예측값 :  [124.59975]
실제값 :  [104.35] 예측값 :  [125.907234]
실제값 :  [104.24] 예측값 :  [128.22816]
실제값 :  [104.69] 예측값 :  [129.21945]
실제값 :  [104.49] 예측값 :  [128.70596]
실제값 :  [104.87] 예측값 :  [127.50587]
실제값 :  [105.05] 예측값 :  [126.95207]
실제값 :  [104.88] 예측값 :  [127.78209]
실제값 :  [104.56] 예측값 :  [129.08382]
실제값 :  [104.81] 예측값 :  [129.49554]
실제값 :  [105.2] 예측값 :  [130.95642]
실제값 :  [105.46] 예측값 :  [131.11827]
실제값 :  [104.87] 예측값 :  [130.42953]
실제값 :  [105.12] 예측값 :  [130.24539]
실제값 :  [105.79] 예측값 :  [130.26262]
실제값 :  [105.8] 예측값 :  [129.30664]
실제값 :  [105.54] 예측값 :  [130.08899]
실제값 :  [104.95] 예측값 :  [129.95294]
실제값 :  [104.71] 예측값 :  [132.16283]
실제값 :  [104.71] 예측값 :  [134.11597]        <- 2020.07월의 CPI 총 지수
'''