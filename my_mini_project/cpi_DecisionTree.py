import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
kf = KFold(n_splits = 5)
scaler = StandardScaler()


## 데이터 불러오기
x = np.load('./my_mini_project/npydata/cpi_train_x.npy')
y = np.load('./my_mini_project/npydata/cpi_train_y.npy')
print(x.shape)      # (208, 5, 13)
print(y.shape)      # (208, 1)

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.1, shuffle = False)
print(x_train.shape)        # (187, 5, 13)
print(x_test.shape)         # (21, 5, 13)
print(y_train.shape)        # (187, 1)
print(y_test.shape)         # (21, 1)

## reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)        # (187, 65)
print(x_test.shape)         # (21, 65)

## 모델 파이프라인 구성
params = {
    'max_depth': [4, 6, 8, 10, 12],
    'criterion': ['mse'],
    'max_leaf_nodes': [10, 20, 30, 40]
}

pipe = Pipeline([('scaler', StandardScaler()), 'dct', DecisionTreeRegressor()])

model = RandomizedSearchCV(pipe, param_distributions = params, cv = KFold)

## 모델 훈련
model.fit(x_train, y_train)

## 평가 및 예측
score = model.score(x_test, y_test)
print("Score : ", score)

x_pred = model.predict(x_test)
for i in range(5):
    print('실제값 : ', y_test[i], '예측값 : ', x_pred[i])