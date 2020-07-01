import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

## 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)            # (60000, 28, 28)
print(x_test.shape)             # (60000,)
print(y_train.shape)            # (10000, 28, 28)
print(y_test.shape)             # (10000,)

## 데이터 전처리
# 정규화
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
print(x_train.shape)            # (60000, 784)
print(x_test.shape)             # (10000, 784)

# 범주화
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)            # (60000, 10)
print(y_test.shape)             # (10000, 10)

## 모델링
model = Sequential()
model.add(Dense(32, input_shape = (28 * 28, ), activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(10, activation = 'softmax'))

model.summary()

## 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy',
                     metrics = ['accuracy'],
                     optimizer = 'adam')
hist = model.fit(x_train, y_train, verbose = 2,
                 epochs = 80, batch_size = 32,
                 validation_split = 0.2, callbacks = [es])

## 모델 평가
res = model.evaluate(x_test, y_test, batch_size = 32)
print("Loss : ", res[0])
print("Accuracy : ", res[1])

## 훈련 과정 시각화
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

plt.figure(figsize = (10, 6))           # 그래프 크기 (10, 6)인치
plt.subplot(2, 1, 1)                    # 2행 1열의 그래프 중 첫 번째 그래프
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('Loss & val_Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper_right')

plt.subplot(2, 1, 2)                    # 2행 1열의 그래프 중 두 번째 그래프
plt.plot(hist.history['accuracy'], marker = '.', c = 'violet', label = 'acc')
plt.plot(hist.history['val_accuracy'], marker = '.', c = 'green', label = 'val_acc')
plt.grid()
plt.title('Accuracy & val_Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc = 'upper_right')
plt.show()