import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPooling2D, Input
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[0])
plt.show()



# (50000, 32, 32, 3)
# (10000, 32, 32, 3)
# (50000, 1)
# (10000, 1)