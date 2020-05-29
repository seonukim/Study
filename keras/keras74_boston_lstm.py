import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

scaler = RobustScaler()
pca = PCA(n_components = 8)
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

# 1. load_Data
x, y = load_boston(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

# 1-1. preprocessing
# 1-1-1. Principle Component Analysis
pca.fit(x)
x = pca.transform(x)
print(x.shape)

# 1-1-2. Scaling
# x = scaler.fit_transform(x)
# print(x.shape)

# 1-1-3. data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 1234)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 1-1-4. train_data reshape
x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)

print(x_train.shape)
print(x_test.shape)


# 2. Modeling_LSTM
model = Sequential()

model.add(LSTM(16, activation = 'relu',
               input_shape = (8, 1),
               return_sequences = True))
model.add(LSTM(16, activation = 'relu',
               return_sequences = True))
model.add(LSTM(32, activation = 'relu',
               return_sequences = True))
model.add(LSTM(32, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

model.summary()


# 3. Compile & Fitting
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 100, batch_size = 32,
                 validation_split = 0.2, verbose = 1)

print(hist.history.keys())


# 4. Evaluate & Predict
res = model.evaluate(x_test, y_test)
print("Result : ", res)

pred = model.predict(x_test)
print("Predict : ", pred)


# 5. R2 score
print("R2 Score : ", r2_score(y_test, pred))
