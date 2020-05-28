import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)


# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 1-1. 정규화
x_train = x_train.reshape(-1, 28, 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28).astype('float32') / 255.0

# 1-2. OHE
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)


# 2. 모델링
model = Sequential()

model.add(Dense(16, input_shape = (28, 28), activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(rate = 0.2))

model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.summary()


# 3. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 100, batch_size = 32,
                 validation_split = 0.05, verbose = 1)

print(hist.history.keys())


# 4. 모델 평가
res = model.evaluate(x_test, y_test, batch_size = 32)
print("Result : ", res)
print("Loss : ", res[0])
print("Accuracy : ", res[1])

'''
Loss :  0.34962257936000823
Accuracy :  0.8790000081062317
'''