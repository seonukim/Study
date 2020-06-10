import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
scaler = MinMaxScaler()
# scaler = StandardScaler()


# 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv',
                    header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv',
                   header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv',
                         header = 0, index_col = 0)

print('train.shape : ', train.shape)                # (10000, 75) : x_train, test
print('test.shape : ', test.shape)                  # (10000, 71) : x_predict
print('submission.shape : ', submission.shape)      # (10000, 4)  : y_predict


# 결측치 확인
print(train.isna().sum())
print(test.isna().sum())

# 결측치 처리
train = train.fillna(train.mean())
test = test.fillna(test.mean())
print("=" * 40)
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

# train = train.interpolate()
# test = test.interpolate()
# print("=" * 40)
# print(train.isnull().sum())
# print(test.isnull().sum())

# 데이터 분할
x = train.iloc[:, :71]
y = train.iloc[:, 71:]
print("=" * 40)
print(x.shape)          # (10000, 71)
print(y.shape)          # (10000, 4)

# 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print("=" * 40)
print(x_train.shape)        # (8000, 71)
print(x_test.shape)         # (2000, 71)
print(y_train.shape)        # (8000, 4)
print(y_test.shape)         # (2000, 4)


# numpy 형 변환
# x_train = x_train.values
# x_test = x_test.values
# y_train = y_train.values
# y_test = y_test.values
# print(type(x_train))
# print(type(x_test))
# print(type(y_train))
# print(type(y_test))         # <class 'numpy.ndarray'>




# 파라미터 선언
param = [
    {'rf__n_estimators': [10, 100, 150], 'rf__criterion': ['mae'], 'rf__max_leaf_nodes': [5, 20, 40]},
    {'rf__n_estimators': [10, 100, 150], 'rf__criterion': ['mse'], 'rf__max_leaf_nodes': [5, 20, 40]}
]

# 파이프라인
pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestRegressor())])

# 랜덤서치 모델 구현
model = RandomizedSearchCV(pipe, param_distributions = param, cv = 5, verbose = 1, n_jobs = -1)

# 모델 훈련
model.fit(x_train, y_train)

# 모델 평가
score = model.score()

# 예측 및 제출 파일 생성
y_pred = model.predict(test)
print('예측값 : \n', y_pred)

y_pred.to_csv('./dacon/submission_200610_1.csv')
