# keras56_mnist_DNN.py 를 땡겨와라
# input_dim = 154로 모델을 만드시오

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, y_train, x_test, y_test를 반환해 준다.

print(x_train[0])
# x의 0번째를 한번본다
print('y_train :',y_train[0])
# y_train : 5
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#(60000, 28, 28)
#(10000, 28, 28)
#(60000,)               # 60000개의 스칼라를 가진 디멘션하나짜리
#(10000,)

print(x_train[0].shape)
plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()

#데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)#(60000,10)

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255


print("x.shape", x_train.shape) 
# x.shape (60000, 28, 28, 1)
print("y.shape", y_train.shape)  
# y.shape (60000, 10)



from sklearn.decomposition import PCA

pca = PCA(n_components = 154)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


print("x.shape", x_train.shape) 
# x.shape (60000, 154)
print("y.shape", y_train.shape)  
# y.shape (60000, 10)


# 2. 모델링
model = Sequential()
model.add(Flatten(input_shape= (154)))   #(9, 9, 10)
model.add(Dense(15, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dropout(0.2))


model.add(Dense(10, activation ='relu')) 
model.add(Dense(15, activation ='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))
model.summary()
print(x_train)
print(y_train)

# 결과값: accuracy: 0.9932

# 3. 훈련

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 15, batch_size = 50, verbose= 2)


loss,acc = model.evaluate(x_test,y_test,batch_size=30)

print(f"loss : {loss}")
print(f"acc : {acc}")
