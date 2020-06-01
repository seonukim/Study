import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10)
ss = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rs = RobustScaler()
pca = PCA(n_components = 15)

# 1. data
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)              # (569, 30)
print(y.shape)              # (569,)

# 1-1. PCA
x = pca.fit_transform(x)
print(x.shape)              # (569, 25)


# 1-2. split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)                # (455, 25)
print(x_test.shape)                 # (114, 25)
print(y_train.shape)                # (455,)
print(y_test.shape)                 # (114,)

# 1-3. preprocessing
x_train = mms.fit_transform(x_train)


# 2. Modeling _ DNN
model = Sequential()

model.add(Dense(32, input_shape = (15, ), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(rate = 0.15))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


# 3. Compile & Fitting
model.compile(loss = 'binary_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 200, batch_size = 1,
                 validation_split = 0.2, verbose = 1)
print(hist.history.keys())


# 4. Evaluate Model
res = model.evaluate(x_test, y_test)
print("=" * 35)
print("Result : ", res)
print("loss : ", res[0])
print("acc : ", res[1])

pred = model.predict(x_test)
pred = pred.reshape(-1, ).astype('int64')
print("=" * 35)
print("Predict of 1 ~ 5 : ", pred[:5])
print("Test data of 1 ~ 5 : ", y_test[:5])


'''
Result
loss :  1685.875239857456
acc :  0.9035087823867798
===================================
Predict of 1 ~ 5 :  [1 0 1 0 0]
'''