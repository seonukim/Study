# download dataset from keras.datasets
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model, Input
from keras.layers import Dense, LSTM, Conv2D, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Data preprocessing 1. one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)

# Data preprocessing 2. normalization
x_train = x_train/255.0
x_test = x_test/255.0

# model

input1 = Input(shape=(32,32,3))
conv2d_1 = Conv2D(8, kernel_size=(5,5), padding='same', activation='relu')(input1)
# dropout_1 = Dropout(rate=0.2)(conv2d_1)
# flatten_1 = Flatten()(dropout_1)
output_1 = Dense(10, activation='sigmoid')(conv2d_1)

model = Model(inputs='input1',outputs='output_1')

# model = Sequential()
# model.add(Conv2D(filters=8, kernel_size=(5,5), padding='same', activation='relu', input_shape=(32,32,3)))
# model.add(Flatten())
# model.add(Dense(10, activation='sigmoid'))

# compile, fit

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, batch_size=1024, epochs=20, verbose=2)

# evaluate

loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(loss)