import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes

es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10)
ss = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rs = RobustScaler()
pca = PCA(n_components = 8)


# 1. load data
x, y = load_diabetes(return_X_y = True)
print(x.shape)
print(y.shape)

# 1-1. preprocessing
x = rs.fit_transform(x)
print(x[:5])

# 1-2 Principle Component Analysis(PCA)
pca.fit(x)
x = pca.transform(x)
print(x.shape)

# 1-3. data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 1-4. data reshape
x_train = x_train.reshape(x_train.shape[0], 8, 1)
x_test = x_test.reshape(x_test.shape[0], 8, 1)
print(x_train.shape)
print(x_test.shape)


# 2. Modeling
# 2-1. Sequential Model
model = Sequential()

model.add(LSTM(128, input_shape = (8, 1),
               activation = 'relu',
               return_sequences = True))
model.add(LSTM(128, activation = 'relu',
               return_sequences = True))
model.add(Dropout(rate = 0.1))
model.add(LSTM(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(1, activation = 'relu'))

model.summary()


# 3. Compile & Fitting
model.compile(loss = 'mean_squared_error',
              metrics = ['mse'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 1000, batch_size = 1,
                 validation_split = 0.05, verbose = 1)
print(hist.history.keys())


# 4. Evaluate Model
res = model.evaluate(x_test, y_test)
print("Result : ", res)
print("loss : ", res[0])
print("mse : ", res[1])

y_predict = model.predict(x_test)
print("Predict of 1 ~ 5 : \n", y_predict)


# 5. Evaluation index
# 5-1. RMSE
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# 5-2. R2 Score
print("R2 Score : ", r2_score(y_test, y_predict))


'''
Result
loss :  3734.434754103757
mse :  3734.4345703125
Predict of 1 ~ 5 : 
 [[125.05877 ]
 [124.1166  ]
 [137.71922 ]
 [126.53372 ]
 [161.64119 ]]
RMSE :  61.11002118568688
R2 Score :  0.4082762988101515
'''