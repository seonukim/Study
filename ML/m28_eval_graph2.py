## 2. 이진분류_breast_canser

import numpy as np
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.datasets import load_breast_cancer
warnings.filterwarnings('ignore')

## 데이터
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

## 모델링
params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [4, 5, 6, 7, 8, 9],
    'criterion': ['reg:linear'],
    'learning_rate': [0.1, 0.01, 0.2, 0.02],
    'nthread': [4, 5, 6],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9]
}

# xgb = XGBClassifier(n_jobs = -1)
# model = RandomizedSearchCV(estimator = xgb,
#                            param_distributions = params,
#                            cv = 5, n_jobs = -1)

model = XGBClassifier(n_estimators = 300,
                     learning_rate = 0.01, n_jobs = -1)

model.fit(x_train, y_train,
          verbose = True,
          eval_metric = ['logloss', 'error'],
          eval_set = [(x_train, y_train),
                      (x_test, y_test)],
          early_stopping_rounds = 20)

results = model.evals_result()
print("Evaluate's Result : ", results)

y_pred = model.predict(x_test)

## Accuracy Score
print("Accuracy Score : ", accuracy_score(y_test, y_pred))

## 그래프, 시각화
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, results['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('Log Loss of Breast_Cancer')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label = 'Train')
ax.plot(x_axis, results['validation_1']['error'], label = 'Test')
ax.legend()
plt.ylabel('Error')
plt.title('Error of Breast_Cancer')
plt.show()