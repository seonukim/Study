import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
leaky = LeakyReLU(alpha = 0.3)
es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)


## 데이터
train = pd.read_csv('C:/Users/bitcamp\Downloads/'
                    '/cpi_train(1975.01 - 2002.09).csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
test = pd.read_csv('C:/Users/bitcamp/Downloads'
                   '/cpi_test(2002.10 - 2020.05).csv',
                   index_col = 0, header = 0,
                   encoding = 'cp949')
print(train.shape)      # (213, 13)
print(test.shape)       # (213, 13)

## NumPy형 변환
train = train.values
test = test.values
print(type(train))      # <class 'numpy.ndarray'>
print(type(test))       # <class 'numpy.ndarray'>

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

x, y = split_xy(train, 5, 1)
print(x.shape)      # (208, 5, 13)
print(y.shape)      # (208, 1)

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = False)
print(x_train.shape)        # (166, 5, 13)
print(x_test.shape)         # (42, 5, 13)
print(y_train.shape)        # (166, 1)
print(y_test.shape)         # (42, 1)

## Dense모델에 넣기 위해 reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (166, 65)
print(x_test.shape)         # (42, 65)

## Scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])
'''
## 모델링
model = Sequential()
model.add(Dense(100, input_shape = (65, ),
                activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(1, activation = 'relu'))

model.summary()

model.save('./my_mini_project/Dense/Dense_model.h5')

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'rmsprop',
              metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 1,
          callbacks = [es], verbose = 1)

## 모델 평가 및 예측
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])            # loss :  1.1412532769498371
print("mse : ", res[1])             # mse :  1.1412532329559326

pred = model.predict(x_test)
for i in range(5):
    print('실제값 : ', y_test[i], '예측값 : ', pred[i])
'''
'''
실제값 :  [65.164] 예측값 :  [66.05616]
실제값 :  [65.055] 예측값 :  [66.54069]
실제값 :  [64.672] 예측값 :  [66.896965]
실제값 :  [64.452] 예측값 :  [66.94108]
실제값 :  [65.111] 예측값 :  [66.851395]
'''

## test 데이터 분리
a, b = split_xy(test, 5, 1)
print(a.shape)      # (208, 5, 13)
print(b.shape)      # (208, 1)

## 데이터 전처리
a_train, a_test, b_train, b_test = train_test_split(
    a, b, test_size = 0.2, shuffle = False)
print(a_train.shape)        # (166, 5, 13)
print(a_test.shape)         # (42, 5, 13)
print(b_train.shape)        # (166, 1)
print(b_test.shape)         # (42, 1)

## Dense모델에 넣기 위해 데이터 reshape
a_train = a_train.reshape(a_train.shape[0], a_train.shape[1] * a_train.shape[2])
a_test = a_test.reshape(a_test.shape[0], a_test.shape[1] * a_test.shape[2])
print(a_train.shape)        # (166, 65)
print(a_test.shape)         # (42, 65)

## Scaling
scaler.fit(a_train)
a_train = scaler.transform(a_train)
a_test = scaler.transform(a_test)
print(a_train[0])

## 모델 불러오기
model = load_model('./my_mini_project/Dense/Dense_model.h5')

model.summary()

## 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(a_train, b_train,
          epochs = 1000, batch_size = 2,
          callbacks = [es], verbose = 1)

## 모델 평가
res = model.evaluate(a_test, b_test)
print("loss : ", res[0])        # loss :  26.6597063654945
print("mse : ", res[1])         # mse :  26.659706115722656

pred_2 = model.predict(a_test)
for i in range(42):
    print('실제값 : ', b_test[i], '예측값 : ', pred_2[i])
'''
실제값 :  [102.64] 예측값 :  [105.10877]
실제값 :  [102.92] 예측값 :  [105.93016]
실제값 :  [102.85] 예측값 :  [106.56531]
실제값 :  [102.72] 예측값 :  [106.543686]
실제값 :  [102.83] 예측값 :  [107.598434]
실제값 :  [102.61] 예측값 :  [108.56856]
실제값 :  [102.78] 예측값 :  [108.38743]
실제값 :  [103.37] 예측값 :  [109.004524]
실제값 :  [103.49] 예측값 :  [110.387726]
실제값 :  [103.39] 예측값 :  [110.835754]
실제값 :  [102.62] 예측값 :  [111.340324]
실제값 :  [102.99] 예측값 :  [111.924095]
실제값 :  [103.42] 예측값 :  [111.57676]
실제값 :  [104.21] 예측값 :  [110.89941]
실제값 :  [104.1] 예측값 :  [110.65243]
실제값 :  [104.29] 예측값 :  [110.05317]
실제값 :  [104.34] 예측값 :  [110.82879]
실제값 :  [104.13] 예측값 :  [111.53039]
실제값 :  [103.93] 예측값 :  [112.151]
실제값 :  [104.85] 예측값 :  [111.760124]
실제값 :  [105.65] 예측값 :  [112.12572]
실제값 :  [105.46] 예측값 :  [111.91189]
실제값 :  [104.71] 예측값 :  [111.7032]
실제값 :  [104.35] 예측값 :  [111.20279]
실제값 :  [104.24] 예측값 :  [110.55778]
실제값 :  [104.69] 예측값 :  [109.36505]
실제값 :  [104.49] 예측값 :  [107.22308]
실제값 :  [104.87] 예측값 :  [104.85908]
실제값 :  [105.05] 예측값 :  [103.84239]
실제값 :  [104.88] 예측값 :  [103.84634]
실제값 :  [104.56] 예측값 :  [103.85918]
실제값 :  [104.81] 예측값 :  [103.73224]
실제값 :  [105.2] 예측값 :  [104.46961]
실제값 :  [105.46] 예측값 :  [104.5829]
실제값 :  [104.87] 예측값 :  [104.55163]
실제값 :  [105.12] 예측값 :  [103.92594]
실제값 :  [105.79] 예측값 :  [103.70546]
실제값 :  [105.8] 예측값 :  [103.69801]
실제값 :  [105.54] 예측값 :  [103.21263]
실제값 :  [104.95] 예측값 :  [102.21549]
실제값 :  [104.71] 예측값 :  [101.95478]
실제값 :  [104.71] 예측값 :  [101.35535]        <- 2020.07월의 CPI 총 지수
'''
