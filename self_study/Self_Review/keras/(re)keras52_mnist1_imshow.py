import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D

## 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)            # (60000, 28, 28)
print(y_train.shape)            # (60000,)
print(x_test.shape)             # (10000, 28, 28)
print(y_test.shape)             # (10000,)

plt.imshow(x_train[0], 'gray')
plt.show()