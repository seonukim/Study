import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
x, y = load_iris(return_X_y = True)
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

# 그리드/랜덤 서치에서 사용할 매개변수
# param = [
#     {"svm__C": [1, 10, 100, 1000], "svm__kernel": ['linear']},
#     {"svm__C": [1, 10, 100, 1000], "svm__kernel": ['rbf'], "svm__gamma": [0.001, 0.0001]},
#     {"svm__C": [1, 10, 100, 1000], "svm__kernel": ['sigmoid'], "svm__gamma": [0.001, 0.0001]}
# ]

# param = [
#     {'C' : [1, 10, 100, 1000], 'kernel' : ['linear']},
#     {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma' : [0.001, 0.0001]},
#     {'C' : [1, 10, 100, 1000], 'kernel' : ['sigmoid'], 'gamma' : [0.001, 0.0001]}
# ]

param = [
    {'rf__n_estimators': [10, 100, 150], 'rf__criterion': ['gini'], 'rf__max_leaf_nodes': [5, 20, 40]},
    {'rf__n_estimators': [10, 100, 150], 'rf__criterion': ['entropy'], 'rf__max_leaf_nodes': [5, 20, 40]}
]

# 2. 모델
pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestClassifier())])
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

model = RandomizedSearchCV(pipe, param_distributions = param, cv = 5)

# 3. 평가 및 예측
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print("최적의 매개 변수 : ", model.best_params_)
print("Accuracy : ", acc)