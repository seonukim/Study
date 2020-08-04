#keras 56_ mnist_DNN.py 복붙

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.layers import Dropout


(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape)  #(60000, 28, 28)
print(x_test.shape)   #(10000, 28, 28)
print(y_train.shape)  #(60000,)#60000개의 스칼라를 가진 디멘션하나짜리
print(y_test.shape)   #(10000,)

 


print(x_train[0].shape)
plt.imshow(x_train[0], 'gray')
 #plt.imshow(x_train[0])
# plt.show()

#데이터 전처리
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)#(60000,10)

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

X = np.append(x_train, x_test, axis = 0)

print(X.shape)  #(70000, 784)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

best_n_components = np.argmax(cumsum >= 0.99)+ 1
print(best_n_components)
