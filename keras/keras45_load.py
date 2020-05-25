# keras40_lstm_split1.py

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
es = EarlyStopping(monitor = 'loss', mode = 'auto', patience = 10)

# 1. 데이터
a = np.array(range(1, 11))
size = 5            # timesteps = 5
print("=" * 40)
print(a.shape)

# LSTM 모델을 완성하시오

# 1-1. 데이터 분할
# 1-1-1. split_x 함수 정의
def split_x(seq, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i + size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

# 1-2. 데이터 a를 분할 후 x와 y로 분할하기
data = split_x(a, size)
print("=" * 40)
print(data)


x = data[:, 0:4]
x = x.reshape(6, 4, 1)
print("=" * 40)
print(x.shape)
print(x)

y = data[:, -1:]
print("=" * 40)
print(y.shape)
print(y)


# 2. 모델 구성
from keras.models import load_model
model_1 = load_model('.\model\save_keras44.h5')


model_1.summary()

# 3. 훈련
model_1.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model_1.fit(x, y, epochs = 1000, batch_size = 1, verbose = 1, callbacks = [es])


# 4. 실행
loss, mse = model_1.evaluate(x, y)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model_1.predict(x, batch_size = 1)
print("=" * 40)
print("y_predict : \n", y_predict)


'''
Result 1)
loss :  0.002710927976295352
mse  :  0.002710927976295352
y_predict :
 [[5.0754447]
  [5.943391 ]
  [6.948891 ]
  [7.9733124]
  [8.987036 ]
  [9.937738 ]]


Result 2)
loss :  8.886436262400821e-05
mse  :  8.886436262400821e-05
y_predict :
 [[ 5.0073833]
  [ 5.9913664]
  [ 7.0102158]
  [ 8.007418 ]
  [ 9.012097 ]
  [10.009922 ]]


Result 3)
loss :  0.015172582119703293
mse  :  0.015172582119703293
y_predict :
 [[5.103075 ]
  [5.9094715]
  [6.9401693]
  [8.013643 ]
  [8.984726 ]
  [9.738817 ]]
'''