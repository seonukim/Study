# k겹 교차 검증
# GridSearchCV
# cifar10 적용

import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)              # (569, 30)
print(y.shape)              # (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2)
print(x_train.shape)        # (455, 30)
print(x_test.shape)         # (114, 30)
print(y_train.shape)        # (455,)
print(y_test.shape)         # (114,)


# 2. KFold_Cross Validation 객체 생성
kf = KFold(n_splits = 5, shuffle = True)


# 3. parameter 정의
params = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'n_estimators': [3, 10], 'bootstrap': [False], 'max_features': [2, 4, 6]}
]


print(params)

model = GridSearchCV(RandomForestClassifier(), param_grid = params, cv = kf, n_jobs = -1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred))
print("parameters : ", model.best_estimator_)
print("best_params : ", model.best_params_)