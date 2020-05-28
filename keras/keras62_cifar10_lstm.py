import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Flatten, Input
from keras.utils import to_categorical
from keras.datasets import cifar10

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 1-1. 정규화
x_train = x_train.reshape(-1, 3, 1024).astype('float32') / 255.0
x_test = x_test.reshape(-1, 3, 1024).astype('float32') / 255.0

# 1-2. 원핫
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)
print(y_train[0])
print(y_test[0])


# 2. 모델링
input1 = Input(shape = (3, 1024))
layer1 = LSTM(16, activation = 'relu', return_sequences = True)(input1)
layer2 = LSTM(32, activation = 'relu')(layer1)
layer3 = Dense(32, activation = 'relu')(layer2)

layer4 = Dropout(rate = 0.25)(layer3)
layer5 = Dense(64, activation = 'relu')(layer4)
layer6 = Dense(128, activation = 'relu')(layer5)
layer7= Dense(10, activation = 'softmax')(layer6)

model = Model(inputs = input1, outputs = layer7)

model.summary()


# 3. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train,
                 epochs = 20, batch_size = 512,
                 validation_split = 0.05, verbose = 1)

print(hist)


# 4. 평가
res = model.evaluate(x_test, y_test, batch_size = 86)
print(res)