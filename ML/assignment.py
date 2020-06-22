import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score


## Data
train = pd.read_csv('C:/Users/dnsrl/Downloads/data/train.csv',
                    index_col = 0, header = 0)
test = pd.read_csv('C:/Users/dnsrl/Downloads/data/test.csv',
                   index_col = 0, header = 0)
print(train.shape)              # (10000, 75)
print(test.shape)               # (10000, 71)

## 결측치 확인
print("=" * 40)
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

## 결측치 처리
# 선형보간법 이용
train = train.interpolate()
test = test.interpolate()
print("=" * 40)
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

# 남아있는 결측치 평균법 이용
train = train.fillna(train.mean())
test = test.fillna(test.mean())
print("=" * 40)
print(train.isnull().sum()[train.isnull().sum().values > 0])        # Series([], dtype: int64)
print(test.isnull().sum()[test.isnull().sum().values > 0])          # Series([], dtype: int64)

## x, y 데이터로 나누기
x = train.iloc[:, :71]
y = train.iloc[:, 71:]
print("=" * 40)
print(x.shape)            # (10000, 71)
print(y.shape)            # (10000, 4)

## Numpy형 변환
x = x.values
y = y.values
test = test.values
print("=" * 40)
print(type(x))
print(type(y))
print(type(test))

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)
print("=" * 40)
print(x_train.shape)            # (8000, 71)
print(x_test.shape)             # (2000, 71)
print(y_train.shape)            # (8000, 4)
print(y_test.shape)             # (2000, 4)

## 모델링
params = {
    'nthread': [4],
    'objective': ['reg:linear'],
    'learning_rate': [0.01, 0.03],
    'max_depth': [3, 5, 7],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'colsample_bylevel': [0.7, 0.8, 0.9],
    'n_estimators': [100, 200, 300]
}

xgb = XGBRegressor(n_jobs = -1)
grid = GridSearchCV(estimator = xgb,
                    param_grid = params,
                    cv = 5, n_jobs = -1)
model = MultiOutputRegressor(estimator = grid)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("Score : ", score)
print("Best Parameters : ", model.estimators_)
# plot_importance(xgb)
# plt.show()

## SelectFromModel
thresholds = np.sort(xgb.feature_importances_)
print(thresholds.shape)
for thresh in thresholds:
    select = SelectFromModel(estimator = xgb,
                             threshold = thresh,
                             prefit = True)
    select_x_train = select.transform(x_train)
    # print(x_train.shape)
    
    select_xgb = XGBRegressor(n_jobs = -1)
    select_grid = GridSearchCV(estimator = xgb,
                               param_grid = params,
                               cv = 5, n_jobs = -1)
    select_model = MultiOutputRegressor(grid)
    select_test = select.transform(test)
    y_pred = select_model.predict(test)
    
    score = r2_score(y, test)
    print("Thresh = %.3f, n = %d, R2 = %.2f%%%" %(thresh, select_x_train.shape[1],
          score * 100.0))