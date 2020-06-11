import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action = 'ignore')

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

## 데이터
x_train = pd.read_csv('./data/dacon/comp3/x_Train.csv',
                      index_col = 0, header = 0)
y_train = pd.read_csv('./data/dacon/comp3/train_target.csv',
                      index_col = 0, header = 0)
pred = pd.read_csv('./data/dacon/comp3/test_features.csv')
print(x_train.shape)        # (2800, 4)
print(y_train.shape)        # (2800, 4)
print(pred.shape)           # (262500, 6)

## 데이터 스플릿
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)        # (2240, 4)
print(x_test.shape)         # (560, 4)
print(y_train.shape)        # (2240, 4)
print(y_test.shape)         # (560, 4)


## 모델링
model = DecisionTreeRegressor(max_depth = 3)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)


## 예측 및 제출 파일 생성
pred = pred.drop('Time', axis = 1)
# print(x_pred.head())

pred = pred.groupby(pred['id']).mean()
# print(x_pred.shape)        # (700, 4)

y_pred = model.predict(pred)
print("pred : \n", y_pred)

y_pred = pd.DataFrame(y_pred,
                      index = range(2800, 3500),
                      columns = ['X', 'Y', 'M', 'V'])
y_pred.to_csv('./data/dacon/comp3/my_submit_200611(3).csv')
