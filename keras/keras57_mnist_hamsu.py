import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Conv2D
from keras.layers import MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-1. 정규화
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

# 1-2. One_Hot_Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)


# 2. 모델링
input1 = Input(shape = (28, 28, 1))
layer1 = Conv2D(filters = 8, kernel_size = (3, 3),
                padding = 'same', activation = 'relu')(input1)
layer2 = Conv2D(filters = 8, kernel_size = (3, 3),
                padding = 'same', activation = 'relu')(layer1)
layer3 = MaxPooling2D(pool_size = (2, 2))(layer2)
layer4 = Dropout(rate = 0.2)(layer3)

layer5 = Conv2D(filters = 16, kernel_size = (3, 3),
                padding = 'same', activation = 'relu')(layer4)
layer6 = Conv2D(filters = 16, kernel_size = (3, 3),
                padding = 'same', activation = 'relu')(layer5)
layer7 = MaxPooling2D(pool_size = (2, 2))(layer6)
layer8 = Dropout(rate = 0.2)(layer7)

output1 = Conv2D(filters = 32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu')(layer8)
output2 = Conv2D(filters = 32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu')(output1)
output3 = MaxPooling2D(pool_size = (2, 2))(output2)
output4 = Dropout(rate = 0.2)(output3)
output5 = Flatten()(output4)
output6 = Dense(10, activation = 'softmax')(output5)

model = Model(inputs = input1, outputs = output6)

model.summary()


# 3. 컴파일 및 실행
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
             optimizer = 'adam')
hist = model.fit(x_train, y_train,
                 epochs = 10, batch_size = 86,
                 validation_split = 0.05, verbose = 1)

print(hist.history.keys())


# 4. 평가 및 예측
res = model.evaluate(x_test, y_test)
print(res)

