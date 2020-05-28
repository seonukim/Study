import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Input
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-1. 정규화
x_train = x_train.reshape(-1, 28, 28) / 255.0
x_test = x_test.reshape(-1, 28, 28) / 255.0

# 1-2. 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 2. 모델링
model = Sequential()

model.add(LSTM(8, input_shape = (28, 28),
               activation = 'relu', return_sequences = True))
model.add(LSTM(16, return_sequences = True))
model.add(LSTM(16))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# 3. 컴파일 및 실행
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train,
                 epochs = 20, batch_size = 1000,
                 validation_split = 0.05, verbose = 1)

print(hist.history.keys())


# 4. 평가
res = model.evaluate(x_test, y_test)
print(res)

plt.plot(hist.history['accuracy'], label = 'acc', ls = '-', marker = 'o')
plt.plot(hist.history['val_accuracy'], label = 'val_acc', ls = '-', marker = 'x')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc = 'best')
plt.show()