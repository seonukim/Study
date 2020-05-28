import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.datasets import fashion_mnist

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)


# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)        # (60000, 28, 28)
print(x_test.shape)         # (10000, 28, 28)
print(y_train.shape)        # (60000,)
print(y_test.shape)         # (10000,)

# 1-1. 정규화
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 1-2. 원핫
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)
print(y_train[0])       # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
print(y_test[0])        # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]


# 2. Sequential() 모델링
model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (3, 3),
                 input_shape = (28, 28, 1), padding = 'same',
                 activation = 'relu'))
model.add(Conv2D(filters = 8, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# 3. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 100, batch_size = 86,
                 validation_split = 0.05, verbose = 1)

print(hist.history.keys()) 


# 4. 모델 평가
res = model.evaluate(x_test, y_test, batch_size = 86)
print("Result : ", res)
print("Loss : ", res[0])
print("Accuracy : ", res[1])


'''
29번째 epoch에서 종료 되었으며, 결과는 아래와 같음
Loss :  0.213752616520226
Accuracy :  0.923799991607666
'''