import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


## Data
train = pd.read_csv('./data/dacon/comp1/train.csv',
                    index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1//test.csv',
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
'''
## x, y 데이터로 나누기
x1 = train.loc[:, '850_src':'990_src']
x2 = train.loc[:, '850_dst':'990_dst']
print("=" * 40)
print(x1.shape)            # (10000, 15)
print(x2.shape)            # (10000, 15)

## Numpy형 변환
x1 = x1.values
x2 = x2.values
print("=" * 40)
print(type(x1))             # <class 'numpy.ndarray'>
print(type(x2))             # <class 'numpy.ndarray'>

x = np.hstack((x1, x2))
y = train.iloc[:, 71:]
print("=" * 40)
print(x.shape)              # (10000, 30)
print(y.shape)              # (10000, 4)

y = y.values
print("=" * 40)
print(type(y))              # <class 'numpy.ndarray'>

x3 = test.loc[:, '850_src':'990_src']
x4 = test.loc[:, '850_dst':'990_dst']
print("=" * 40)
print(x3.shape)            # (10000, 15)
print(x4.shape)            # (10000, 15)

## Numpy형 변환
x3 = x3.values
x4 = x4.values
print("=" * 40)
print(type(x3))             # <class 'numpy.ndarray'>
print(type(x4))             # <class 'numpy.ndarray'>

test = np.hstack((x3, x4))
print("=" * 40)
print(type(test))
'''
## x, y 데이터로 나누기
x = train.iloc[:, :71]
y = train.iloc[:, 71:]
print("=" * 40)
print(x.shape)              # (10000, 71)
print(y.shape)              # (10000, 4)

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
print(x_train.shape)            # (8000, 30)
print(x_test.shape)             # (2000, 30)
print(y_train.shape)            # (8000, 4)
print(y_test.shape)             # (2000, 4)

## feature importance
xgb = XGBRegressor()
model = MultiOutputRegressor(xgb)
model.fit(x_train, y_train)
print(len(model.estimators_))           # 4

print(model.estimators_[0].feature_importances_)
print(model.estimators_[1].feature_importances_)
print(model.estimators_[2].feature_importances_)
print(model.estimators_[3].feature_importances_)
print("Score : ", model.score(x_test, y_test))

# params = {
#     'nthread': [5],
#     'n_estimators': [100, 150, 300, 450],
#     'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
#     'colsample_bylevel': [0.6, 0.7, 0.8, 0.9],
#     'max_depth': [4, 6, 8, 10],
#     'learning_rate': [0.1, 0.01, 0.2, 0.02]
# }

## 모델링
for i in range(len(model.estimators_)):
    threshold = np.sort(model.estimators_[i].feature_importances_)

    for j in threshold:
        select = SelectFromModel(model.estimators_[i],
                                 threshold = j,
                                 prefit = True)
        
        # search = GridSearchCV(estimator = XGBRegressor(n_jobs = 6),
        #                       param_grid = params,
        #                       cv = 5)
        
        select_x_train = select.transform(x_train)
        select_x_test = select.transform(x_test)

        search_model = MultiOutputRegressor(XGBRegressor(n_estimators = 300,
                                                         colsample_bytree = 0.9,
                                                         colsample_bylevel = 0.9,
                                                         max_depth = 10,
                                                         learning_rate = 0.1,
                                                         nthread = 12,
                                                         n_jobs = 6))
        
        search_model.fit(select_x_train, y_train)

        # Predict
        y_pred = search_model.predict(select_x_test)
        mae = mean_absolute_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)

        print("Thresh=%.3f, n = %d, R2 : %.2f%%, MAE : %.3f"%(j, select_x_train.shape[1], score * 100.0, mae))

        select_x_pred = select.transform(test)
        pred = search_model.predict(select_x_pred)

        # submission
        a = np.arange(10000, 20000)
        submission = pd.DataFrame(pred, a)
        submission.to_csv('./data/dacon/comp1/sub_XG%i_%.5f.csv' %(i, mae),
                          index = True,
                          header = ['hhb', 'hbo2', 'ca', 'na'],
                          index_label = 'id')
                          