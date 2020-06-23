## 1. LGBM으로 boston 데이터 만들기

import numpy as np
import pandas as pd
import warnings

from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
warnings.filterwarnings('ignore')

## data
x, y = load_boston(return_X_y = True)
print(x.shape)              # (506, 13)
print(y.shape)              # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)
print(x_train.shape)        # (404, 13)
print(x_test.shape)         # (102, 13)
print(y_train.shape)        # (404,)
print(y_test.shape)         # (102,)

## 모델 _ 1
# params = {
#     'learning_rate': [0.01, 0.1, 0.02, 0.2, 0.03, 0.3],
#     'max_depth': [5, 6, 7, 8, 9, 10],
#     'n_estimators': [100, 150, 300, 450, 600],
# }

model = LGBMRegressor(n_jobs = -1,
                      learning_rate = 0.01,
                      max_depth = -1,
                      n_estimators = 500)

# model = RandomizedSearchCV(estimator = lgb,
#                            param_distributions = params,
#                            cv = 5, n_jobs = -1)

## 훈련 예측
model.fit(x_train, y_train,
          verbose = True,
          eval_metric = ['logloss', 'rmse'],
          eval_set = [(x_train, y_train),
                      (x_test, y_test)],
          early_stopping_rounds = 100)

results = model.evals_result_
print("Evaluate's Result : ", results)


## SFM 모델링
thresholds = np.sort(model.feature_importances_)
print(thresholds)

for i in thresholds:
    select = SelectFromModel(estimator = model,
                             threshold = i,
                             prefit = True)
    
    select_x_train = select.transform(x_train)
    select_x_test = select.transform(x_test)

    select_model = LGBMRegressor(n_jobs = -1,
                                 n_estimators = 500,
                                 max_depth = -1,
                                 learning_rate = 0.01,
                                 colsample_bytree = 0.7)
    select_model.fit(select_x_train, y_train,
                     verbose = False,
                     eval_metric = ['logloss', 'rmse'],
                     eval_set = [(select_x_train, y_train),
                                 (select_x_test, y_test)],
                     early_stopping_rounds = 100)
    
    y_pred = select_model.predict(select_x_test)
    r2 = r2_score(y_test, y_pred)
    print("Thresh = %.3f, n = %d, R2 = %.2f%%"
          %(i, select_x_train.shape[1], r2 * 100.0))

'''
Thresh = 0.000, n = 13, R2 = 92.24%
Thresh = 36.000, n = 12, R2 = 92.26%
Thresh = 207.000, n = 11, R2 = 92.11%
Thresh = 219.000, n = 10, R2 = 91.75%
Thresh = 469.000, n = 9, R2 = 91.82%
Thresh = 487.000, n = 8, R2 = 91.30%
Thresh = 494.000, n = 7, R2 = 90.72%
Thresh = 648.000, n = 6, R2 = 89.68%
Thresh = 692.000, n = 5, R2 = 88.05%
Thresh = 845.000, n = 4, R2 = 86.04%
Thresh = 1086.000, n = 3, R2 = 86.04%
Thresh = 1183.000, n = 2, R2 = 83.53%
Thresh = 1226.000, n = 1, R2 = 69.51%
'''