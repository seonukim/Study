# keras 56_ mnist_DNN.py 복붙

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.layers import Dropout


(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)
print(y_train.shape)  # (60000,)         # 60000개의 스칼라를 가진 디멘션하나짜리
print(y_test.shape)   # (10000,)

 


print(x_train[0].shape)
plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()

# 데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                # (60000, 10)

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
# reshape로 4차원을 만들어준 다음에(CNN모델을 집어넣기 위해서, 가로 새로 채널), 
# astype은 파일을 변환해준다는 것이다 원래는 정수형태로(각 픽셀마다 255중에 255면 완전 찐한 검정색)데이터가 0부터 255까지가 들어가 있는데
# 정수형태로 들어가 있다. 하지만 우리가 넣으려고 하는 minmax는 0부터 1까지인 실수이기때문에 float를 정수에서 실수로 바꾸어주게 된다. 
# 나누기 255는 0부터 1까지로 나누어주기위해서 그 사이를 255개로 쪼개 주는 것이다. 이것이 정규화이다.  
# 255로 나누게 되면 최댓값이 1이 되고 최소값이 0이 된다. 
# print(x_train)
# print(x_test)
# print(y_train.shape)
# print(y_test.shape)


print("x.shape", x_train.shape) 
# x.shape (60000, 28, 28, 1)
print("y.shape", y_train.shape)  
# y.shape (60000, 10)


# 2. 모델링
input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs = 50, batch_size = 256, 
                    validation_split = 0.2)

decoded_imgs = autoencoder.predict(x_test)


import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n) :
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# print(x_train)
# print(y_train)

# #결과값: accuracy: 0.9932

# #3. 훈련

# model.compile(loss = 'categorical_crossentropy',
#               optimizer = 'adam', metrics = ['accuracy'])
# model.fit(x_train, y_train, epochs = 15, batch_size = 50, verbose= 2)


# loss,acc = model.evaluate(x_test,y_test,batch_size=30)

# print(f"loss : {loss}")
# print(f"acc : {acc}")
