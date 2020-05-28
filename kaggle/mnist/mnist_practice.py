import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

pd.read_csv()
# 1. load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


'''
x_train = x_train / 255.0
test = test / 255.0

x_train = x_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, num_classes = 10)
print(y_train.shape)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size = 0.1,
    shuffle = True, random_state = 2)

print(x_train.shape)
print(y_train.shape)
print(test.shape)


# 2. CNN Modeling
model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = (3, 3),
                 padding = 'same', input_shape = (28, 28, 1),
                 activation = 'relu'))
model.add(Conv2D(filters = 8, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(rate = 0.25))

model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(rate = 0.25))

model.add(Conv2D(filters = 32, kernel_size = (5, 5),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (5, 5),
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(rate = 0.25))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 3. compile & fitting
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'], optimizer = 'adam')
hist = model.fit(x_train, y_train,
                 epochs = 5, batch_size = 512,
                 validation_split = 0.05, verbose = 1)
print(hist.history.keys())
'''

