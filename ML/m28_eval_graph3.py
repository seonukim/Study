## 3. 다중분류_iris

import numpy as np
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris

## 데이터
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

## 모델링
model = XGBClassifier(n_estimators = 1000,
                      learning_rate = 0.01,
                      objective = 'multi:softmax',
                      n_jobs = -1)

model.fit(x_train, y_train,
          verbose = True,
          eval_metric = ['mlogloss', 'merror'],
          eval_set = [(x_train, y_train),
                      (x_test, y_test)],
          early_stopping_rounds = 100)

results = model.evals_result()
print("Evaluate's Result : ", results)

y_pred = model.predict(x_test)

## Accuracy Score
print("Accuracy Score : ", accuracy_score(y_test, y_pred))

## 그래프, 시각화
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label = 'Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('Log Loss of Iris')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label = 'Train')
ax.plot(x_axis, results['validation_1']['merror'], label = 'Test')
ax.legend()
plt.ylabel('Error')
plt.title('Error of Iris')
plt.show()