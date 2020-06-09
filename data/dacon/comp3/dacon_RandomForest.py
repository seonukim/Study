import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping
scaler = StandardScaler()
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

x_train = np.sqrt(x_train.groupby(x_train['id']).mean())
print(x_train.shape)        # (2800, 4)


x_train = pd.read_csv('./data/dacon/comp3/x_Train.csv',
                      index_col = 0, header = 0,
                      encoding = 'utf-8')
print(x_train.head())
print(x_train.shape)        # (2800, 4)

print(y_train.head())
print(y_train.shape)        # (2800, 4)

print(x_train.isna().sum())

'''
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

# x_train = x_train.reshape(-1, 4, 1)
# x_test = x_test.reshape(-1, 4, 1)
# print(x_train.shape)        # (2240, 4, 1)
# print(x_test.shape)         # (560, 4, 1)


### 모델
kf = KFold(n_splits = 5, shuffle = True)
params = {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8], 
          'n_estimators': [3, 10], 'bootstrap': [False], 'max_features': [2, 4, 6]}


print(params)

model = RandomizedSearchCV(RandomForestRegressor(), param_distributions = params, cv = kf, n_jobs = -1)

### 훈련
model.fit(x_train, y_train)

### 예측
x_pred = x_pred.drop('Time', axis = 1)
# print(x_pred.head())

x_pred = np.sqrt(x_pred.groupby(x_pred['id']).mean())
# print(x_pred.shape)        # (700, 4)

x_pred = x_pred.values
x_pred = scaler.fit_transform(x_pred)
# x_pred = x_pred.reshape(-1, 4, 1)
# print(type(x_pred))         # <class 'numpy.ndarray'>

y_predict = model.predict(x_pred)
print("Predict : \n", y_predict)

submit = pd.DataFrame(y_predict)
print(submit.head())

submit.to_csv('./data/dacon/comp3/mysubmit_2.csv')
'''