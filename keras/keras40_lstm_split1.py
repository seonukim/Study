# keras40_lstm_split1.py

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

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

# x_predict = 



# 2. 모델 구성
'''
model = Sequential()
model.add(LSTM(10, activation = 'relu',
               input_length = 4, input_dim = 1))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(1))

model.summary()
'''

input1 = Input(shape = (4, 1))
dense1 = LSTM(10, activation = 'relu', return_sequences = True)(input1)
dense2 = LSTM(10, activation = 'relu', return_sequences = True)(dense1)
dense3 = LSTM(8, activation = 'relu', return_sequences = True)(dense2)
dense4 = LSTM(8, activation = 'relu', return_sequences = False)(dense3)

output1 = Dense(8)(dense4)
output2 = Dense(7)(output1)
output3 = Dense(1)(output2)

model = Model(inputs = input1, outputs = output3)

model.summary()


# 3. 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x, y, epochs = 1000, batch_size = 1, verbose = 1, callbacks = [es])

# 4. 실행
y_predict = model.predict(x, batch_size = 1)
print(y_predict)