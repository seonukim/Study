import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.layers import MaxPooling2D, Dropout, Conv2D
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 1-1. 정규화
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0

# 1-2. 원핫인코딩
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)
print(y_train[0])
print(y_test[0])


# 2. 모델링
input1 = Input(shape = (32, 32, 3))
layer1 = Dense(16, activation = 'relu')(input1)
layer2 = Dense(16, activation = 'relu')(layer1)
layer3 = Dropout(rate = 0.25)(layer2)

layer4 = Dense(32, activation = 'relu')(layer3)
layer5 = Dense(32, activation = 'relu')(layer4)
layer6 = Dropout(rate = 0.25)(layer5)

layer7 = Flatten()(layer6)
layer8 = Dense(64, activation = 'relu')(layer7)
layer9 = Dense(64, activation = 'relu')(layer8)
layer10 = Dense(10, activation = 'softmax')(layer9)

model = Model(inputs = input1, outputs = layer10)

model.summary()


# 3. 컴파일 및 실행
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 100, batch_size = 512,
                 validation_split = 0.05, verbose = 1)

print(hist.history.keys())


# 4. 평가 및 예측
res = model.evaluate(x_test, y_test, batch_size = 512)
print(res)