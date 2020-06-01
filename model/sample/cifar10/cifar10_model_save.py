import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

modelpath = './model/sample/cifar10/check-{epoch:02d}-{loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'loss',
                     save_best_only = True, save_weights_only = False)
pca = PCA(n_components = 5)
mms = MinMaxScaler()


# 1. data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)                    # (50000, 32, 32, 3)
print(x_test.shape)                     # (10000, 32, 32, 3)
print(y_train.shape)                    # (50000, 1)
print(y_test.shape)                     # (10000, 1)

# 1-1. normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# 1-2. ohe
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                    # (50000, 10)
print(y_test.shape)                     # (10000, 10)


# 2. model
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.1))

model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.15))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# 3. compile fit
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
model.fit(x_train, y_train, callbacks = [es, cp],
          epochs = 20, batch_size = 86, verbose = 3,
          validation_split = 0.2)

model.save('./model/sample/cifar10/cifar10_model_save.h5')
model.save_weights('./model/sample/cifar10/cifar10_weight_save.h5')


# 4. evaluate
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])
print("acc : ", res[1])