import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D
from keras.utils import to_categorical
from keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# 2. 데이터 전처리
# 2-1. 정규화
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
# x_train = x_train / 255.0
# x_test = x_test / 255.0
print(x_train[0])

# 2-2. One_Hot_Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[0].shape)


# 3. 모델링
model = Sequential()

model.add(Dense(8, input_shape = (28 * 28, ), activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(rate = 0.4))

model.add(Dense(32, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(rate = 0.4))

# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(rate = 0.25))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(rate = 0.4))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# 4. 컴파일 및 실행
model.compile(optimizer = 'adam',
              metrics = ['accuracy'],
              loss = 'binary_crossentropy')
hist = model.fit(x_train, y_train,
                 epochs = 5, batch_size = 86,
                 validation_split = 0.01, verbose = 1)
print(hist.history.keys())


# 5. 평가 및 예측
res = model.evaluate(x_test, y_test)
print(res)
