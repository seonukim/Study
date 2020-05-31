import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense
from keras.layers import Dropout, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
pca = PCA(n_components = 15)
mms = MinMaxScaler()

# 1. data
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)              # (569, 30)
print(y.shape)              # (569,)

# 1-1. pca
x = pca.fit_transform(x)

# 1-2. split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

# 1-3. scaling
x_train = mms.fit_transform(x_train)
x_train = x_train.reshape(-1, 15, 1, 1)
x_test = x_test.reshape(-1, 15, 1, 1)

# 2. model
model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = (3, 3),
                 input_shape = (15, 1, 1), padding = 'same',
                 activation = 'relu'))
model.add(Conv2D(filters = 8, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


# 3. fit
model.compile(loss = 'binary_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
model.fit(x_train, y_train, callbacks = [es],
          epochs = 20, batch_size = 1,
          validation_split = 0.05)


# 4. evaluate
res = model.evaluate(x_test, y_test)
print("Result : ", res)
print("loss : ", res[0])
print("acc : ", res[1])

pred = model.predict(x_test)
pred = pred.reshape(-1, ).astype('int64')
print("Predict : ", pred[:5])
print("test data : ", y_test[:5])