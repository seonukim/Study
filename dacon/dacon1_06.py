import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action = 'ignore')

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from keras.models import Input, Model
from keras.layers import LSTM, Dense, Dropout
from keras.layers import LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
kfold = KFold(n_splits = 5)
leaky = LeakyReLU(alpha = 0.3)


## 데이터 불러오기
train = pd.read_csv('./data/dacon/comp1/train.csv',
                    index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv',
                   index_col = 0, header = 0)
submit = pd.read_csv('./data/dacon/comp1/sample_submission.csv',
                     index_col = 0, header = 0)
print(train.shape)          # (10000, 75)
print(test.shape)           # (10000, 71)
print(submit.shape)         # (10000, 4)

## 결측치 확인
print("=" * 40)
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

## 선형보간법을 이용한 결측치 처리
train = train.interpolate()
test = test.interpolate()
print("=" * 40)
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

## 남은 결측치는 평균법 사용하여 처리
train = train.fillna(train.mean())
test = test.fillna(train.mean())
print("=" * 40)
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

## x, y 데이터로 나누기
x_train = train.iloc[:, :71]
y_train = train.iloc[:, 71:]
print(x_train.shape)        # (10000, 71)
print(y_train.shape)        # (10000, 4)

## LSTM을 위한 차원 스케일 조정
x_train = x_train.values
y_train = y_train.values
# x_train = x_train.reshape(-1, 71, 1)
print(x_train.shape)
print(type(x_train))

## 모델링
def mymodel():
    input1 = Input(shape = (71, ))
    x = Dense(32, activation = leaky)(input1)
    x = Dense(32, activation = leaky)(x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(16, activation = leaky)(x)
    x = Dropout(rate = 0.1)(x)
    output = Dense(4, activation = leaky)(x)
    model = Model(inputs = input1, outputs = output)
    model.compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = ['mse'])
    return model

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    epochs = [20, 40, 60, 80, 100]
    return {'batch_size': batches,
            'epochs': epochs}
            
params = create_hyperparameter()

## 케라스로 RandomizedSearchCV 모델 구성
model = KerasRegressor(build_fn = mymodel, verbose = 1)
search = RandomizedSearchCV(model, param_distributions = params,
                            n_jobs = -1, cv = kfold)

search.fit(x_train, y_train)

print(search.best_params_)
print(search.best_score_)

## 예측 및 제출 파일 생성
pred = search.predict(test)
pred = pd.DataFrame(pred)

pred.to_csv('./dacon/mysubmission_200615.csv',
            index = range(10000, 20000),
            columns = ['hhb', 'hbo2', 'ca', 'na'])