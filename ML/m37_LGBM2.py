## 2. LGBM으로 iris 데이터 만들기

import numpy as np
import pandas as pd
import warnings

from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
pca = PCA(n_components = 1)

## data
x, y = load_iris(return_X_y = True)
print(x.shape)              # (150, 4)
print(y.shape)              # (150,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)
print(x_train.shape)        # (120, 4)
print(x_test.shape)         # (30, 4)
print(y_train.shape)        # (120,)
print(y_test.shape)         # (30,)

## 모델_1
model = LGBMClassifier(n_jobs = -1,
                       learning_rate = 0.01,
                       max_depth = -1,
                       n_estimators = 500,
                       objective = 'multiclass',
                       metric = ['multi_logloss', 'multi_error'])

## 훈련 예측
model.fit(x_train, y_train,
          verbose = True,
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

    select_model = LGBMClassifier(n_jobs = -1,
                                  learning_rate = 0.01,
                                  max_depth = -1,
                                  n_estimators = 500,
                                  objective = 'multiclass',
                                  metric = ['multi_logloss', 'multi_error'])
    
    select_model.fit(select_x_train, y_train,
                     verbose = False,
                     eval_set = [(select_x_train, y_train),
                                 (select_x_test, y_test)],
                     early_stopping_rounds = 100)
    
    y_pred = select_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
    print("Thresh = %.3f, n = %d, ACC = %.2f%%"
          %(i, select_x_train.shape[1], acc * 100.0))

'''
Thresh = 3.000, n = 4, ACC = 93.33%
Thresh = 5.000, n = 3, ACC = 93.33%
Thresh = 8.000, n = 2, ACC = 93.33%
Thresh = 9.000, n = 1, ACC = 93.33%
'''