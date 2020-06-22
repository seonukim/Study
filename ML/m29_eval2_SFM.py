# 2. 이진분류_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer
warnings.filterwarnings('ignore')

## 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)          # (569, 30)
print(y.shape)          # (569,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)
print(x_train.shape)        # (455, 30)
print(x_test.shape)         # (114, 30)
print(y_train.shape)        # (455,)
print(y_test.shape)         # (114,)

## 모델링
model = XGBClassifier(n_estimators = 1000,
                      learning_rate = 0.01,
                      n_jobs = 6)

model.fit(x_train, y_train,
          verbose = True,
          eval_metric = ['logloss', 'error'],
          eval_set = [(x_train, y_train),
                      (x_test, y_test)],
          early_stopping_rounds = 100)

results = model.evals_result()
print("Evaluate's Result : ", results)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for i in thresholds:
    select = SelectFromModel(estimator = model,
                             threshold = i,
                             prefit = True)
    select_x_train = select.transform(x_train)
    select_x_test = select.transform(x_test)

    select_model = XGBClassifier(n_estimators = 1000,
                                 learning_rate = 0.01,
                                 n_jobs = 6)
    select_model.fit(select_x_train, y_train,
                     verbose = True,
                     eval_metric = ['logloss', 'rmse'],
                     eval_set = [(select_x_train, y_train),
                                 (select_x_test, y_test)],
                     early_stopping_rounds = 100)
    y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, y_pred)

    print("Thresh = %.3f, n = %d, acc : %.2f%%"
          %(i, select_x_train.shape[1], score * 100.0))

# acc : 0.9649122807017544