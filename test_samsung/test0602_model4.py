import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.models import Sequential, Model

pca = PCA(n_components = 1)
ss = StandardScaler()

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
# print(samsung.shape)            # (509, 1)
# print(hite.shape)               # (509, 5)

samsung = samsung.reshape(samsung.shape[0], )
# print(samsung.shape)            # (509,)

samsung = split_x(samsung, size)
# print(samsung.shape)            # (504, 6)

x1 = samsung[:, 0:5]
y1 = samsung[:, 5]
# print(x1.shape)                 # (504, 5)
# print(y1.shape)                 # (504,)

x2 = hite[5:510, :]
# print(x2.shape)                 # (504, 5)


x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, test_size = 0.2)
print("=" * 40)
print(x1_train.shape)           # (403, 5)
print(x1_test.shape)            # (101, 5)
print(y1_train.shape)           # (403,)
print(y1_test.shape)            # (101,)

x2_train, x2_test = train_test_split(
    x2, test_size = 0.2)
print("=" * 40)
print(x2_train.shape)           # (403, 5)
print(x2_test.shape)            # (101, 5)


# x1_train = pca.fit_transform(x1_train)
# x1_test = pca.fit_transform(x1_test)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.fit_transform(x2_test)
print("=" * 40)
print(x1_train.shape)           # (403, 5)
print(x1_test.shape)            # (101, 5)
print(x2_train.shape)           # (403, 1)
print(x2_test.shape)            # (101, 1)

x1_train = ss.fit_transform(x1_train)
x1_test = ss.fit_transform(x1_test)
x2_train = ss.fit_transform(x2_train)
x2_test = ss.fit_transform(x2_test)
# print(x1_train)

x1_train = x1_train.reshape(-1, 5, 1)
x1_test = x1_test.reshape(-1, 5, 1)
x2_train = x2_train.reshape(x2_train.shape[0], 1, 1)
x2_test = x2_test.reshape(-1, 1, 1)
print("=" * 40)
print(x1_train.shape)           # (403, 5, 1)
print(x1_test.shape)            # (101, 5, 1)
print(x2_train.shape)           # (403, 1, 1)
print(x2_test.shape)            # (101, 1, 1)


# 2. 모델구성
input1 = Input(shape = (5, 1))
x1 = LSTM(32, activation = 'relu')(input1)
x1 = Dense(32, activation = 'relu')(x1)
x1 = Dropout(rate = 0.2)(x1)

input2 = Input(shape = (1, 1))
x2 = LSTM(16, activation = 'relu')(input2)
x2 = Dense(16, activation = 'relu')(x2)
x2 = Dropout(rate = 0.2)(x2)

merge = concatenate([x2, x2])

output = Dense(1, activation = 'relu')(merge)

model = Model(inputs = [input1, input2],
              outputs = output)

model.summary()


# 3. 컴파일 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
hist = model.fit([x1_train, x2_train], y1_train,
                 epochs = 200, batch_size = 50,
                 validation_split = 0.05, verbose = 1)


# 4. 평가 및 예측
res = model.evaluate([x1_test, x2_test], y1_test)
print("=" * 40)
print(res)


y_predict = model.predict([x1_test, x2_test])
print(y_predict[:5])


import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))
plt.plot(hist.history['loss'], c = 'red', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue', marker = '.')
plt.ylabel("loss")
plt.xlabel("epoch")
# plt.legend(loc = 'lower left')
plt.legend(['loss', 'val_loss'], loc = 'upper right')
plt.show()