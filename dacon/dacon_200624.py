import numpy as np
import pandas as pd
import warnings ; warnings.filterwarnings('ignore')
import time
import tqdm
import os

path = 'D:\Study\data\dacon'
os.chdir(path)

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor

## 데이터 불러오기
train = pd.read_csv(path + '/comp1/train.csv',
                    index_col = 'id', encoding = 'utf-8')
test = pd.read_csv(path + '/comp1/test.csv',
                   index_col = 'id', encoding = 'utf-8')
submission = pd.read_csv(path + '/comp1/sample_submission.csv',
                         index_col = 'id', encoding = 'utf-8')
print(train.shape, '\n', test.shape, '\n', submission.shape)

## 교차검증 함수 정의
def model_scoring_cv(estimator, x_data, y_data, cv = 5):
    start = time.time()
    score = cross_val_score(estimator, x_data, y_data, cv = cv,
                            scoring = 'neg_mean_absolute_error').mean()
    stop = time.time()
    print(f'Validation Time : {round(stop - start, 3)} sec')
    return score

## 데이터
feature_names = list(test)
target_names = list(submission)

x_train = train[feature_names]
x_test = test[feature_names]
print(x_train.shape, '\n', x_test.shape)
# (10000, 71), (10000, 71)

y_train = train[target_names]
y_train_1 = y_train['hhb']
y_train_2 = y_train['hbo2']
y_train_3 = y_train['ca']
y_train_4 = y_train['na']

## 모델 정의
N_ESTIMATORS = 1000
NUM_LEAVES = 100
LEARNING_RATE = 0.05
COLSAMPLE_BYTREE = 0.7
SUBSAMPLE = 0.8
REG_ALPHA = 5
REG_LAMBDA = 7

# 하이퍼 파라미터
params = {
    'n_estimators': N_ESTIMATORS,
    'num_leaves': NUM_LEAVES,
    'learning_rate': LEARNING_RATE,
    'colsample_bytree': COLSAMPLE_BYTREE,
    'subsample': SUBSAMPLE,
    'reg_alpha': REG_ALPHA,
    'reg_lambda': REG_LAMBDA
}

# 모델 생성
lgb = LGBMRegressor(objective = 'l1', subsample_freq = 1,
                    silent = False, random_state = 18,
                    importance_type = 'gain',
                    n_jobs = -1, **params)

# 모델 래핑
multi_lgb = MultiOutputRegressor(estimator = lgb, n_jobs = -1)

## 결측치 처리해보기
src_list = ['650_src', '660_src', '670_src', '680_src', '690_src',
            '700_src', '710_src', '720_src', '730_src', '740_src',
            '750_src', '760_src', '770_src', '780_src', '790_src',
            '800_src', '810_src', '820_src', '830_src', '840_src',
            '850_src', '860_src', '870_src', '880_src', '890_src',
            '900_src', '910_src', '920_src', '930_src', '940_src',
            '950_src', '960_src', '970_src', '980_src', '990_src']

dst_list = ['650_dst', '660_dst', '670_dst', '680_dst', '690_dst',
            '700_dst', '710_dst', '720_dst', '730_dst', '740_dst',
            '750_dst', '760_dst', '770_dst', '780_dst', '790_dst',
            '800_dst', '810_dst', '820_dst', '830_dst', '840_dst',
            '850_dst', '860_dst', '870_dst', '880_dst', '890_dst',
            '900_dst', '910_dst', '920_dst', '930_dst', '940_dst',
            '950_dst', '960_dst', '970_dst', '980_dst', '990_dst']

# print(model_scoring_cv(estimator = multi_lgb,
#                        x_data = x_train.fillna(0),
#                        y_data = y_train))

## 결측치 처리
alpha = x_train[dst_list]
beta = x_test[dst_list]

for i in tqdm.tqdm(x_train.index):
    alpha.loc[i] = alpha.loc[i].interpolate()

for i in tqdm.tqdm(x_test.index):
    beta.loc[i] = beta.loc[i].interpolate()

print(alpha.isnull().sum())
print(beta.isnull().sum())

alpha.loc[alpha['700_dst'].isnull(),'700_dst'] = alpha.loc[alpha['700_dst'].isnull(),'710_dst']
alpha.loc[alpha['690_dst'].isnull(),'690_dst'] = alpha.loc[alpha['690_dst'].isnull(),'700_dst']
alpha.loc[alpha['680_dst'].isnull(),'680_dst'] = alpha.loc[alpha['680_dst'].isnull(),'690_dst']
alpha.loc[alpha['670_dst'].isnull(),'670_dst'] = alpha.loc[alpha['670_dst'].isnull(),'680_dst']
alpha.loc[alpha['660_dst'].isnull(),'660_dst'] = alpha.loc[alpha['660_dst'].isnull(),'670_dst']
alpha.loc[alpha['650_dst'].isnull(),'650_dst'] = alpha.loc[alpha['650_dst'].isnull(),'660_dst']

beta.loc[beta['700_dst'].isnull(),'700_dst'] = beta.loc[beta['700_dst'].isnull(),'710_dst']
beta.loc[beta['690_dst'].isnull(),'690_dst'] = beta.loc[beta['690_dst'].isnull(),'700_dst']
beta.loc[beta['680_dst'].isnull(),'680_dst'] = beta.loc[beta['680_dst'].isnull(),'690_dst']
beta.loc[beta['670_dst'].isnull(),'670_dst'] = beta.loc[beta['670_dst'].isnull(),'680_dst']
beta.loc[beta['660_dst'].isnull(),'660_dst'] = beta.loc[beta['660_dst'].isnull(),'670_dst']
beta.loc[beta['650_dst'].isnull(),'650_dst'] = beta.loc[beta['650_dst'].isnull(),'660_dst']

x_train[dst_list] = np.array(alpha)
x_test[dst_list] = np.array(beta)
print(model_scoring_cv(estimator = multi_lgb,
                       x_data = x_train,
                       y_data = y_train))