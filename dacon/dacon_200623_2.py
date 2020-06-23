import os
path = 'D:/Study/data/dacon'
os.chdir(path)
# print(os.listdir(path))

import pandas as pd
import numpy as np
import time
import warnings ; warnings.filterwarnings('ignore')

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

## data
train = pd.read_csv(path + '/comp1/train.csv', index_col = 0)
test = pd.read_csv(path + '/comp1/test.csv', index_col = 0)
submission = pd.read_csv(path + '/comp1/sample_submission.csv', index_col = 0)
print(train.shape, test.shape, submission.shape)
# (10000, 75) (10000, 71) (10000, 4)

feature_names = list(test)
target_names = list(submission)

x_train = train[feature_names]
x_test = test[feature_names]
print(x_train.shape, x_test.shape)
# (10000, 71) (10000, 71)

y_train = train[target_names]
y_train_1 = y_train['hhb']
y_train_2 = y_train['hbo2']
y_train_3 = y_train['ca']
y_train_4 = y_train['na']
print(y_train.shape)        # (10000, 4)


# base model 구성
params = {
    'n_estimators': 1200,
    'num_leaves': 100,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'subsample': 0.9,
    'reg_alpha': 5,
    'reg_lambda': 7
}

base_model = LGBMRegressor(objective = 'l1', subsample_freq = 1,
                           silent = False, random_state = 66,
                           importance_type = 'gain', **params,
                           n_jobs = -1)

multi_model = MultiOutputRegressor(estimator = base_model, n_jobs = -1)

## 교차검증 함수 정의
def model_scoring_cv(model, x, y, cv = 10):
    start = time.time()
    score = cross_val_score(model, x, y, cv = cv,
                            scoring = 'neg_mean_absolute_error').mean()
    stop = time.time()
    print(f'validation Time : {round(stop - start, 3)} sec')
    return score

src_list = ['650_src', '660_src', '670_src', '680_src', '690_src', '700_src', '710_src', '720_src', '730_src', 
            '740_src', '750_src', '760_src', '770_src', '780_src', '790_src', '800_src', '810_src', '820_src', 
            '830_src', '840_src', '850_src', '860_src', '870_src', '880_src', '890_src', '900_src', '910_src', 
            '920_src', '930_src', '940_src', '950_src', '960_src', '970_src', '980_src', '990_src']

dst_list = ['650_dst', '660_dst', '670_dst', '680_dst', '690_dst', '700_dst', '710_dst', '720_dst', '730_dst', 
            '740_dst', '750_dst', '760_dst', '770_dst', '780_dst', '790_dst', '800_dst', '810_dst', '820_dst', 
            '830_dst', '840_dst', '850_dst', '860_dst', '870_dst', '880_dst', '890_dst', '900_dst', '910_dst', 
            '920_dst', '930_dst', '940_dst', '950_dst', '960_dst', '970_dst', '980_dst', '990_dst']

# model_scoring_cv(multi_model, x_train.fillna(-1), y_train)

## 결측치 처리
alpha = x_train[dst_list]
beta = x_test[dst_list]

for i in tqdm(x_train.index):
    alpha.loc[i] = alpha.loc[i].interpolate()

for i in tqdm(x_test.index):
    beta.loc[i] = beta.loc[i].interpolate()

print(alpha.isnull().sum(), '\n', beta.isnull().sum())

alpha.loc[alpha['700_dst'].isnull(),'700_dst']=alpha.loc[alpha['700_dst'].isnull(),'710_dst']
alpha.loc[alpha['690_dst'].isnull(),'690_dst']=alpha.loc[alpha['690_dst'].isnull(),'700_dst']
alpha.loc[alpha['680_dst'].isnull(),'680_dst']=alpha.loc[alpha['680_dst'].isnull(),'690_dst']
alpha.loc[alpha['670_dst'].isnull(),'670_dst']=alpha.loc[alpha['670_dst'].isnull(),'680_dst']
alpha.loc[alpha['660_dst'].isnull(),'660_dst']=alpha.loc[alpha['660_dst'].isnull(),'670_dst']
alpha.loc[alpha['650_dst'].isnull(),'650_dst']=alpha.loc[alpha['650_dst'].isnull(),'660_dst']

beta.loc[beta['700_dst'].isnull(),'700_dst']=beta.loc[beta['700_dst'].isnull(),'710_dst']
beta.loc[beta['690_dst'].isnull(),'690_dst']=beta.loc[beta['690_dst'].isnull(),'700_dst']
beta.loc[beta['680_dst'].isnull(),'680_dst']=beta.loc[beta['680_dst'].isnull(),'690_dst']
beta.loc[beta['670_dst'].isnull(),'670_dst']=beta.loc[beta['670_dst'].isnull(),'680_dst']
beta.loc[beta['660_dst'].isnull(),'660_dst']=beta.loc[beta['660_dst'].isnull(),'670_dst']
beta.loc[beta['650_dst'].isnull(),'650_dst']=beta.loc[beta['650_dst'].isnull(),'660_dst']

x_train[dst_list] = np.array(alpha)
x_test[dst_list] = np.array(beta)
# model_scoring_cv(multi_model, x_train, y_train)

## rho; 측정 거리(mm)
for col in dst_list:
    x_train[col] *= (x_train['rho'] ** 2)
    x_test[col] *= (x_test['rho'] ** 2)

gap_feature_names = []
for i in range(650, 1000, 10):
    gap_feature_names.append(str(i) + '_gap')

alpha = pd.DataFrame(np.array(x_train[src_list]) - np.array(x_train[dst_list]),
                     columns = gap_feature_names, index = train.index)
beta = pd.DataFrame(np.array(x_test[src_list]) - np.array(x_test[dst_list]),
                     columns = gap_feature_names, index = test.index)

x_train = pd.concat((x_train, alpha), axis = 1)
x_test = pd.concat((x_test, beta), axis = 1)

print(x_train.shape, y_train.shape, x_test.shape)
model_scoring_cv(multi_model, x_train, y_train)