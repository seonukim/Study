import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar100
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

modelpath = './model/sample/cifar100/check-{epoch:02d}-{loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'loss',
                     save_best_only = True, save_weights_only = False)
pca = PCA(n_components = 5)
mms = MinMaxScaler()


# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
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
print(y_train.shape)                    # (50000, 100)
print(y_test.shape)                     # (10000, 100)


# 2. model
model = Sequential()

model.add(Conv2D(filters = 256, kernel_size = (5, 5),
                 padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
model.add(Dropout(rate = 0.2))
model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(MaxPooling2D(pool_size = (5, 5)))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(MaxPooling2D(pool_size = (5, 5)))

model.add(Flatten())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))

model.summary()


# 3. compile fit
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
model.fit(x_train, y_train, callbacks = [es, cp],
          epochs = 20, batch_size = 512, verbose = 1,
          validation_split = 0.05)

model.save('./model/sample/cifar100/cifar100_model_save.h5')
model.save_weights('./model/sample/cifar100/cifar100_weight_save.h5')


# 4. evaluate
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])
print("acc : ", res[1])
