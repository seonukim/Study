import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
scaler = RobustScaler()
warnings.filterwarnings(action = 'ignore')

# 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv',
                    index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv',
                   index_col = 0, header = 0)
submit = pd.read_csv('./data/dacon/comp1/sample_submission.csv',
                     index_col = 0, header = 0)
print(train.shape)
print(test.shape)
print(submit.shape)

# 결측치 확인
print("=" * 40)
print(train.isnull().sum())
print(test.isnull().sum())

# 결측치 처리
train = train.fillna(train.mean())
test = test.fillna(test.mean())
print("=" * 40)
print(train.isnull().sum())
print(test.isnull().sum())

# 데이터 나누기
x = train.iloc[:, :71]
y = train.iloc[:, 71:]
print(x.shape)      # (10000, 71)
print(y.shape)      # (10000, 4)

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)    # (8000, 71)
print(x_test.shape)     # (2000, 71)
print(y_train.shape)    # (8000, 4)
print(y_test.shape)     # (2000, 4)

# 스케일링
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
submit = scaler.fit_transform(submit)
print(type(x_train))    # <class 'numpy.ndarray'>
print(type(x_test))     # <class 'numpy.ndarray'>
print(type(submit))     # <class 'numpy.ndarray'>

y_train = y_train.values
y_test = y_test.values
# y_train 분류
y_train_1 = y_train[:, 0]
y_train_2 = y_train[:, 1]
y_train_3 = y_train[:, 2]
y_train_4 = y_train[:, 3]

y_test_1 = y_test[:, 0]
y_test_2 = y_test[:, 1]
y_test_3 = y_test[:, 2]
y_test_4 = y_test[:, 3]

# 모델링
model = GradientBoostingRegressor()

# 모델 훈련
model.fit(x_train, y_train_1)
score = model.score(x_test, y_test_1)
print("score1 : ", score)
print(model.feature_importances_)
y_pred_1 = model.predict(test)

model.fit(x_train, y_train_2)
score = model.score(x_test, y_test_2)
print("score2 : ", score)
print(model.feature_importances_)
y_pred_2 = model.predict(test)

model.fit(x_train, y_train_3)
score = model.score(x_test, y_test_3)
print("score3 : ", score)
print(model.feature_importances_)
y_pred_3 = model.predict(test)

model.fit(x_train, y_train_4)
score = model.score(x_test, y_test_4)
print("score4 : ", score)
print(model.feature_importances_)
y_pred_4 = model.predict(test)

# 모델 평가
# score = model.score(x_test, y_test)

# 결과 출력
print(model.feature_importances_)
# print(score)

'''
### DecisionTreeRegressor ###

[0.20587324 0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.01274717 0.         0.         0.         0.04795241 0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.0094543      
 0.         0.         0.00834683 0.         0.         0.2387007      
 0.         0.         0.         0.         0.1679802  0.
 0.14274724 0.156536   0.00151323 0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.00814867 0.         0.         0.        ]

-0.07827178673370114


### RandomForestRegressor ###

[0.04200695 0.00819293 0.00763657 0.00710432 0.0070164  0.00727816     
 0.00732804 0.00753617 0.00721881 0.00858313 0.0079611  0.00873184     
 0.00955112 0.0106769  0.01200443 0.01466854 0.0159268  0.01482792     
 0.01260821 0.0126389  0.01069975 0.0106954  0.00917561 0.00889935     
 0.00839418 0.00860531 0.0082889  0.00788614 0.00789633 0.00798567     
 0.00820347 0.00819246 0.0085366  0.00846675 0.00859378 0.00991117     
 0.00945299 0.00898371 0.01027893 0.01041031 0.01082127 0.01092187     
 0.0118063  0.01121707 0.01260404 0.01277239 0.01194206 0.02747459     
 0.02056263 0.04373925 0.02794068 0.03086608 0.04336665 0.02049182     
 0.02879342 0.02162436 0.01871221 0.01666486 0.01762455 0.01281945     
 0.01373111 0.01520912 0.01485083 0.01453563 0.01127945 0.01236737     
 0.01734469 0.02016898 0.02766259 0.01644929 0.01458133]

0.0906309834219274


### GradientBoostingRegressor ###
'''