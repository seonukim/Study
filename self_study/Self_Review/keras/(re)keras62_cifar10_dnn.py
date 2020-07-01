import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Input
from keras.layers import Conv2D, Flatten, Dropout
from keras.layers import Dense, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10
path = 'C:/Users/bitcamp/Desktop/서누/Review/Model/'
os.chdir(path)

model_path = path + 'DNN-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath = model_path, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto')
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

## 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)                # (50000, 32, 32, 3)
print(x_test.shape)                 # (10000, 32, 32, 3)
print(y_train.shape)                # (50000, 1)
print(y_test.shape)                 # (10000, 1)

## 정규화
x_train = x_train.reshape(-1, 32 * 32 * 3).astype('float32') / 255      # Dense 모델은 2차원 와꾸를 받는다
x_test = x_test.reshape(-1, 32 * 32 * 3).astype('float32') / 255        # Dense 모델은 2차원 와꾸를 받는다
print(x_train.shape)                # (50000, 3072)
print(x_test.shape)                 # (10000, 3072)

## 레이블 범주화
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                # (50000, 10)
print(y_test.shape)                 # (10000, 10)

## 모델링
input1 = Input(shape = (3072, ))
x1 = Dense(32, activation = 'relu')(input1)
x1 = Dense(32, activation = 'relu')(x1)
x1 = Dropout(rate = 0.2)(x1)
x1 = Dense(16, activation = 'relu')(x1)
x1 = Dense(16, activation = 'relu')(x1)
x1 = Dropout(rate = 0.2)(x1)
x1 = Dense(10, activation = 'softmax')(x1)

model = Model(inputs = input1, outputs = x1)
model.summary()

## 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
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
plt.plot(hist.history['accuracy'], marker = '.', c = 'violet', label = 'acc')
plt.plot(hist.history['val_accuracy'], marker = '.', c = 'green', label = 'val_acc')
plt.grid()
plt.title('Acc & val_Acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.ylim(0, 1)
plt.legend(loc = 'best')
plt.show()