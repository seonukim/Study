import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import LSTM, Flatten, Dropout
from keras.layers import Dense, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
path = 'C:/Users/bitcamp/Desktop/서누/Review/Model/'
os.chdir(path)

model_path = path + 'LSTM-{epoch:02d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)
cp = ModelCheckpoint(filepath = model_path, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto')

## 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)

## 데이터 정규화
x_train = x_train / 255
x_test = x_test / 255
print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)

## 레이블 범주화
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                # (60000, 10)
print(y_test.shape)                 # (10000, 10)

## 모델링
input1 = Input(shape = (28, 28))
x1 = LSTM(32, activation = 'relu', return_sequences = True)(input1)
x1 = LSTM(32, activation = 'relu')(x1)
x1 = Dropout(rate = 0.2)(x1)
x1 = Dense(16, activation = 'relu')(x1)
x1 = Dense(16, activation = 'relu')(x1)
x1 = Dropout(rate = 0.2)(x1)
output1 = Dense(10, activation = 'softmax')(x1)

model = Model(inputs = input1, outputs = output1)
model.summary()

## 컴파일 및 훈련
model.compile(optimizer = 'adam', metrics = ['acc'], loss = 'categorical_crossentropy')
hist = model.fit(x_train, y_train, verbose = 1,
                 epochs = 100, batch_size = 128,
                 validation_split = 0.2, callbacks = [es, cp])
print(hist.history.keys())

## 모델 평가
res = model.evaluate(x_test, y_test, batch_size = 128)
print('loss : ', res[0])
print('acc : ', res[1])

## 훈련 과정 시각화
plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('Loss & val_Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'best')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = '.', c = 'violet', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.', c = 'green', label = 'val_acc')
plt.grid()
plt.title('Acc & val_Acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc = 'best')
plt.show()