import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
x, y = load_iris(return_X_y = True)
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

# 2. 모델
model = SVC()
pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])

pipe.fit(x_train, y_train)
print('acc : ', pipe.score(x_test, y_test))
# acc :  0.8666666666666667