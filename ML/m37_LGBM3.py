## 3. LGBM으로 cancer 데이터 만들기

import numpy as np
import pandas as pd
import warnings

from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
pca = PCA(n_components = 1)

## data
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)              # (569, 30)
print(y.shape)              # (569,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)
print(x_train.shape)        # (455, 30)
print(x_test.shape)         # (114, 30)
print(y_train.shape)        # (455,)
print(y_test.shape)         # (114,)

## 모델_1
model = LGBMClassifier(n_estimators = 1000,
                       max_depth = 20,
                       learning_rate = 0.01,
                       colsample_bytree = 0.7,
                       n_jobs = -1)

## 훈련 예측
model.fit(x_train, y_train,
          verbose = True,
          eval_metric = ['logloss', 'error'],
          eval_set = [(x_train, y_train),
                      (x_test, y_test)],
          early_stopping_rounds = 100)

res = model.evals_result_
print(res)

## 모델링
thresholds = np.sort(model.feature_importances_)

for i in thresholds:
    select = SelectFromModel(estimator = model,
                             threshold = i,
                             prefit = True)
    select_x_train = select.transform(x_train)
    select_x_test = select.transform(x_test)

    select_model = LGBMClassifier(n_estimators = 1000,
                                  max_depth = -1,
                                  learning_rate = 0.01,
                                  colsample_bytree = 0.7,
                                  n_jobs = -1)
    
    select_model.fit(select_x_train, y_train,
                     verbose = False,
                     eval_set = [(select_x_train, y_train),
                                 (select_x_test, y_test)],
                     eval_metric = ['logloss', 'error'],
                     early_stopping_rounds = 100)
    
    y_pred = select_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
    print("Thresh = %.2f, n = %d, ACC = %.3f%%"
          %(i, select_x_train.shape[1], acc * 100.0))

'''
Thresh = 82.00, n = 30, ACC = 97.368%
Thresh = 141.00, n = 29, ACC = 97.368%
Thresh = 164.00, n = 28, ACC = 96.491%
Thresh = 224.00, n = 27, ACC = 97.368%
Thresh = 228.00, n = 26, ACC = 97.368%
Thresh = 235.00, n = 25, ACC = 95.614%
Thresh = 245.00, n = 24, ACC = 97.368%
Thresh = 255.00, n = 23, ACC = 96.491%
Thresh = 260.00, n = 22, ACC = 97.368%
Thresh = 268.00, n = 21, ACC = 96.491%
Thresh = 287.00, n = 20, ACC = 96.491%
Thresh = 301.00, n = 19, ACC = 96.491%
Thresh = 328.00, n = 18, ACC = 95.614%
Thresh = 339.00, n = 17, ACC = 94.737%
Thresh = 342.00, n = 16, ACC = 94.737%
Thresh = 346.00, n = 15, ACC = 95.614%
Thresh = 358.00, n = 14, ACC = 94.737%
Thresh = 376.00, n = 13, ACC = 94.737%
Thresh = 414.00, n = 12, ACC = 96.491%
Thresh = 441.00, n = 11, ACC = 94.737%
Thresh = 442.00, n = 10, ACC = 95.614%
Thresh = 451.00, n = 9, ACC = 95.614%
Thresh = 517.00, n = 8, ACC = 94.737%
Thresh = 522.00, n = 7, ACC = 94.737%
Thresh = 533.00, n = 6, ACC = 92.105%
Thresh = 535.00, n = 5, ACC = 92.982%
Thresh = 605.00, n = 4, ACC = 92.105%
Thresh = 673.00, n = 3, ACC = 92.982%
Thresh = 778.00, n = 2, ACC = 91.228%
Thresh = 1098.00, n = 1, ACC = 72.807%
'''