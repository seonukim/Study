import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.layers import MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.datasets import cifar10

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 1-1. 정규화
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 1-2. 원핫인코딩
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)
# print(y_train.shape)
# print(y_test.shape)


# 2. 모델링
input1 = Input(shape = (32, 32, 3))
layer1 = Conv2D(filters = 16, kernel_size = (5, 5),
                padding = 'same', activation = 'relu')(input1)
# layer2 = Conv2D(filters = 8, kernel_size = (3, 3),
#                 padding = 'same', activation = 'relu')(layer1)
layer3 = MaxPooling2D(pool_size = (2, 2))(layer1)
layer4 = Dropout(rate = 0.2)(layer3)

layer5 = Conv2D(filters = 32, kernel_size = (5, 5),
                padding = 'same', activation = 'relu')(layer4)
# layer6 = Conv2D(filters = 16, kernel_size = (3 ,3),
#                 padding = 'same', activation = 'relu')(layer5)
layer7 = MaxPooling2D(pool_size = (2, 2))(layer5)
layer8 = Dropout(rate = 0.2)(layer7)

output1 = Flatten()(layer8)
output2 = Dense(64, activation = 'relu')(output1)
# output3 = Dense(64, activation = 'relu')(output2)
output4 = Dense(10, activation = 'softmax')(output2)

model = Model(inputs = input1, outputs = output4)

model.summary()


# 3. 컴파일 및 실행
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train,
                 epochs = 20, batch_size = 86,
                 validation_split = 0.05, verbose = 1)

print(hist.history.keys())


# 4. 평가 및 예측
res = model.evaluate(x_test, y_test)
print("res : ", res)


pred = model.predict(x_test)
y_test = np.argmax(y_test, axis = 1)
pred = np.argmax(pred, axis = 1)
print("y_test : \n", y_test[:10])
print("pred : \n", pred[:10])