import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.models import Sequential, Model



def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)

size = 6


# 1. 데이터
# npy 불러오기
samsung = np.load('./data/samsung.npy', allow_pickle = 'True') # 객체 배열(object)를 저장할 수 있게 해줌
hite = np.load('./data/hite.npy', allow_pickle = 'True') # object는 str, int와 같은 데이터 타입

print(samsung.shape)        # (509, 1)
print(hite.shape)           # (509, 5)

samsung = samsung.reshape(samsung.shape[0], )
print(samsung.shape)        # (509, )

samsung = split_x(samsung, size)
print(samsung.shape)        # (504, 6)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print(x_sam.shape)          # (504, 5)
print(y_sam.shape)          # (504, )

x_hit = hite[5:510, :]
print(x_hit.shape)          # (504, 5)

x_hit = x_hit.reshape(x_hit.shape[0], 5, 1)
print(x_hit.shape)          # (504, 5, 1)

x_sam = x_sam.reshape(x_sam.shape[0], 5, 1)
print(x_sam.shape)          # (504, 5, 1)


#2. 모델구성 
input1 = Input(shape = (5, 1))
x1 = LSTM(64)(input1)
x1 = Dense(10)(x1)
x1 = Dropout(rate = 0.2)(x1)

input2 = Input(shape = (5, 1))
x2 = LSTM(64)(input2)
x2 = Dense(5)(x2)
x2 = Dropout(rate = 0.2)(x2)

merge = Concatenate()([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1, input2], output = output)

model.summary()


# 3. 컴파일, 훈련 
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x_sam, x_hit], y_sam, epochs = 5)
