import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()

leaky = LeakyReLU(alpha = 0.2)
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10)

### 데이터 ###
x = pd.read_csv('./data/dacon/comp3/train_features.csv',
                encoding = 'utf-8')
y = pd.read_csv('./data/dacon/comp3/train_target.csv',
                index_col = 0, header = 0,
                encoding = 'utf-8')
x_pred = pd.read_csv('./data/dacon/comp3/test_features.csv',
                     encoding = 'utf-8')
print(x.shape)                # (1050000, 6)
print(y.shape)                # (2800, 5)
print(x_pred.shape)           # (262500, 6)

x_train = x
y_train = y
print(x_train.shape)        # (1050000, 6)
print(y_train.shape)        # (2800, 5)
print(x_train.head())


x_train = x_train.drop('Time', axis = 1)
print(x_train.head())


x_train = np.power(x_train.groupby(x_train['id']).mean(), 2)
print(x_train.shape)        # (2800, 4)

# x_train.to_csv('./data/dacon/comp3/x_train2.csv')


x_train = pd.read_csv('./data/dacon/comp3/x_train2.csv',
                      index_col = 0, header = 0,
                      encoding = 'utf-8')
print(x_train.head())
print(x_train.shape)        # (2800, 4)

print(y_train.head())
print(y_train.shape)        # (2800, 4)


x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size = 0.2)
print(x_train.shape)        # (2240, 4)
print(x_test.shape)         # (560, 4)
print(y_train.shape)        # (2240, 4)
print(y_test.shape)         # (560, 4)

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 4, 1)
x_test = x_test.reshape(-1, 4, 1)
print(x_train.shape)        # (2240, 4, 1)
print(x_test.shape)         # (560, 4, 1)


### 모델
model = Sequential()
model.add(Conv1D(filters = 10, kernel_size = 5,
                 input_shape = (4, 1), padding = 'same',
                 activation = leaky))
model.add(Conv1D(filters = 7, kernel_size = 3,
                 padding = 'same', activation = leaky))
model.add(MaxPooling1D())
model.add(Dropout(rate = 0.1))
model.add(Conv1D(filters = 16, kernel_size = 3,
                 padding = 'same', activation = leaky))
model.add(Conv1D(filters = 8, kernel_size = 3,
                 padding = 'same', activation = leaky))
model.add(MaxPooling1D())
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(4, activation = leaky))

model.summary()


### 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(x_train, y_train, epochs = 10, callbacks = [es],
          batch_size = 2, validation_split = 0.2)


### 모델 평가
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])
print("mse : ", res[1])


# x_pred = x_pred.drop('Time', axis = 1)
# print(x_pred.head())

# x_pred = np.power(x_pred.groupby(x_pred['id']).mean(), 2)
# print(x_pred.shape)        # (700, 4)

# x_pred.to_csv('./data/dacon/comp3/x_pred.csv')

x_pred = pd.read_csv('./data/dacon/comp3/x_pred.csv',
                     index_col = 0, header = 0)
print(x_pred.shape)         # (700, 4)
print(x_pred.head())

x_pred = x_pred.values
x_pred = scaler.fit_transform(x_pred)
x_pred = x_pred.reshape(-1, 4, 1)
print(type(x_pred))         # <class 'numpy.ndarray'>

y_predict = model.predict(x_pred)
print("Predict : \n", y_predict)


submit = pd.DataFrame(y_predict)
print(submit.head())

submit.to_csv('./data/dacon/comp3/mysubmit_4.csv')
