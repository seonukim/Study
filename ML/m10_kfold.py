# k겹 교차 검증
# KFold, cross_val_score

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
iris = pd.read_csv('./data/csv/iris.csv',
                   header = 0, sep = ',')

x = iris.iloc[:, :3]
y = iris.iloc[:, -1:]
print(x)
print(y)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size = 0.2)

kfold = KFold(n_splits = 5, shuffle = True)
'''
n_splits = 5 ; 데이터를 5등분한다 -> 한 조각 당 20%
'''





allAlgorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    scores = cross_val_score(model, x, y, cv = kfold)

    print(name, '의 정답률 : ')
    print(scores)
