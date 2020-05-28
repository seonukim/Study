import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler      # (x - 최소) / (최대 - 최소)
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print("y_train : ", y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(x_train[0].shape)
# plt.imshow(x_train[0], 'gray')          # plt.imshow() 함수는 데이터의 이미지를 보여준다.
# plt.imshow(x_train[0])
# plt.show()

# 데이터 전처리 - 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 - 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0
'''
1. 데이터를 CNN 모델에 넣기 위해 reshape해서 4차원으로 바꿔준다. (채널 1 추가)
2. astype('float32')는 현재 정수형인 데이터를 실수형으로 바꿔준다.
3. / 255는 정규화를 의미한다. (MinMaxScaler와 거의 동일)
'''

# 모델 구성
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape = (28, 28, 1), activation = 'relu'))
model.add(Conv2D(32, (5, 5), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(rate = 0.25))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2))
model.add(Dropout(rate = 0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, epochs = 5, batch_size = 86)

# 평가 및 예측
res = model.evaluate(x_test, y_test)
print("res : ", res)

# print(hist.history.keys())
# plt.plot(hist.history['accuracy'], label = 'acc', ls = '-', marker = 'o')
# plt.plot(hist.history['loss'], label = 'loss', ls = '-', marker = 'x')
# plt.ylabel('accuracy & loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'best')
# plt.show()

# 0.993399977684021
# 0.9927999973297119
# 0.9922999739646912
# 0.9914000034332275
# 0.9922000169754028
# 0.9912999868392944
# 0.9943000078201294
