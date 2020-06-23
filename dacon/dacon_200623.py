import pandas as pd
import numpy as np
import warnings ; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

## data
train = pd.read_csv('./data/dacon/comp1/train.csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
test = pd.read_csv('./data/dacon/comp1/test.csv',
                   index_col = 0, header = 0,
                   encoding = 'cp949')
print(train.shape)              # (10000, 75)
print(test.shape)               # (10000, 71)

## 결측치 확인
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

## 보간법 사용하여 결측치 처리
train = train.interpolate()
test = test.interpolate()
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

## 나머지 결측치 처리하기
train = train.fillna(-1)
test = test.fillna(-1)
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

## 데이터 나누기
x = train.iloc[:, :71]
y = train.iloc[:, 71:]
print(x.shape)                  # (10000, 71)
print(y.shape)                  # (10000, 4)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = False)
print(x_train.shape)            # (8000, 71)
print(x_test.shape)             # (2000, 71)
print(y_train.shape)            # (8000, 4)
print(y_test.shape)             # (2000, 4)

## 모델링
lgb = LGBMRegressor(n_estimators = 1500, num_leaves = 90,
                    learning_rate = 0.01, colsample_bytree = 0.9,
                    subsample = 1, n_jobs = -1,
                    reg_alpha = 5, reg_lambda = 7,
                    max_depth = -1, random_state = 1,
                    objective = 'l1')

model = MultiOutputRegressor(estimator = lgb,
                             n_jobs = -1)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print(mae)