# keras95를 불러와서 모델을 완성하시오
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
rs = RobustScaler()
mms = MinMaxScaler()
ss = StandardScaler()
mas = MaxAbsScaler()
pca = PCA(n_components = 2)

datasets = np.load('./data/csv/iris.npy')
print(datasets.shape)

x = datasets[:, :]
y = datasets[:, -1:]
print(x.shape)
print(y.shape)

y = y.reshape(-1, )
print(y.shape)


# 1-1. preprocessing
pca.fit(x)
x = pca.transform(x)

# 1-2. normalization
x = mas.fit_transform(x)

# 1-3. data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

print(x_train.shape)            # (120, 4)
print(x_test.shape)             # (30, 4)
print(y_train.shape)            # (120,)
print(y_test.shape)             # (30,)

# 1-4. One Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train[0])
print(y_test[0])


# 2. Modeling _ DNN

# 2-1. Sequential Model
model = Sequential()

model.add(Dense(128, input_shape = (2, ),
                activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 3. compile & fitting
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 1000, batch_size = 1,
                 validation_split = 0.05, verbose = 1)

print(hist.history.keys())


# 4. Evaluate Model
res = model.evaluate(x_test, y_test)
print("=" * 35)
print("Result : ", res)
print("loss : ", res[0])
print("acc : ", res[1])

pred = model.predict(x_test)
pred = np.argmax(pred, axis = 1)
print("=" * 35)
print("Predict of 1 ~ 5 : \n", pred[:5])

y_test = np.argmax(y_test, axis = 1)
print("Test data of 1 ~ 5 : \n", y_test[:5])
