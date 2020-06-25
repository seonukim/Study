## Dacon 코드 공유; 푸리에변환 코드 이해해보기

import os
import pandas as pd
import numpy as np
import warnings
import time
import tqdm

path = 'D:/Study/data/dacon'
os.chdir(path)
warnings.filterwarnings('ignore')

from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

## Data
train = pd.read_csv(path + '/comp1/train.csv',
                    index_col = 'id', encoding = 'utf-8')
test = pd.read_csv(path + '/comp1/test.csv',
                   index_col = 'id', encoding = 'utf-8')
submission = pd.read_csv(path + '/comp1/sample_submission.csv',
                         index_col = 'id', encoding = 'utf-8')
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


## 모델 구성
params = {
    'objective': 'mae',
    'learning_rate': 0.01,
    'max_depth': 15,
    'n_estimators': 1000,
    'colsubsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 5,
    'reg_lambda': 7
}

lgb = LGBMRegressor(num_leaves = 90, importance_type = 'gain',
                    silent = False, subsample_freq = 1, n_jobs = -1,
                    random_state = 77, **params)

multi_lgb = MultiOutputRegressor(estimator = lgb, n_jobs = -1)

## 교차검증 함수 정의
def model_scoring_cv(estimator, x_data, y_data, cv = 10):
    start = time.time()
    score = cross_val_score(estimator, x_data, y_data, cv = cv,
                            scoring = 'neg_mean_absolute_error').mean()
    stop = time.time()
    print(f'Validation Time : {round(stop - start, 3)} sec')
    return score

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
    alpha.loc[i] = alpha.loc[i].interpolate(limit_direction = 'backward')

for i in tqdm.tqdm(x_test.index):
    beta.loc[i] = beta.loc[i].interpolate(limit_direction = 'backward')
print(alpha.isnull().sum())
print(beta.isnull().sum())

for i in tqdm.tqdm(alpha.index):
    alpha.loc[i] = alpha.loc[i].fillna(alpha.mean())

for i in tqdm.tqdm(beta.index):
    beta.loc[i] = beta.loc[i].fillna(beta.mean())
print(alpha.isnull().sum())
print(beta.isnull().sum())

x_train[dst_list] = np.array(alpha)
x_test[dst_list] = np.array(beta)
# print(model_scoring_cv(estimator = multi_lgb,
#                        x_data = x_train,
#                        y_data = y_train))
'''
Validation Time : 299.048 sec
-1.2123732505801688
'''

## rho : 측정 거리(nm)
for col in dst_list:
    x_train[col] *= (x_train['rho'] ** 2)
    x_test[col] *= (x_test['rho'] ** 2)
'''
측정거리(rho)
측정된 빛의 세기는 거리 제곱에 반비례
측정 거리를 제곱해서 곱해준다
'''

# src - dst 차이
gap_feature_names = []
for i in range(650, 1000, 10):
    gap_feature_names.append(str(i) + '_gap')

alpha = pd.DataFrame(np.array(x_train[src_list]) - np.array(x_train[dst_list]),
                     columns = gap_feature_names, index = train.index)
beta = pd.DataFrame(np.array(x_test[src_list]) - np.array(x_test[dst_list]),
                    columns = gap_feature_names, index = test.index)

x_train = pd.concat((x_train, alpha), axis = 1)
x_test = pd.concat((x_test, beta), axis = 1)
print(x_train.shape, x_test.shape, y_train.shape)
# print(model_scoring_cv(estimator = multi_lgb,
#                        x_data = x_train,
#                        y_data = y_train))
'''
적외선 분광분석법에는 적외선 흡수를 통한 분광분석법이 이루어지므로
원래 빛과 측정 빛의 강도 차이를 파장별로 계산해서 파생변수 추가
'''
'''
Validation Time : 453.0 sec
-1.2096001915805936
'''

epsilon = 1e-10

for dst_col, src_col in zip(dst_list, src_list):
    dst_val = x_train[dst_col]
    src_val = x_train[src_col] + epsilon
    delta_ratio = dst_val / src_val
    x_train[dst_col + '_' + src_col + '_ratio'] = delta_ratio

    dst_val = x_test[dst_col]
    src_val = x_test[src_col] + epsilon

    delta_ratio = dst_val / src_val
    x_test[dst_col + '_' + src_col + '_ratio'] = delta_ratio

print(x_train.shape, x_test.shape)
# print(model_scoring_cv(estimator = multi_lgb,
#                        x_data = x_train,
#                        y_data = y_train))
'''
Validation Time : 562.764 sec
-1.135245044443266
'''

## DFT with numpy
alpha_real = x_train[dst_list]
alpha_imag = x_train[dst_list]
beta_real = x_test[dst_list]
beta_imag = x_test[dst_list]

for i in tqdm.tqdm(alpha_real.index):
    alpha_real.loc[i] = alpha_real.loc[i] - alpha_real.loc[i].mean()
    alpha_imag.loc[i] = alpha_imag.loc[i] - alpha_imag.loc[i].mean()

    alpha_real.loc[i] = np.fft.fft(alpha_real.loc[i], norm = 'ortho').real
    alpha_imag.loc[i] = np.fft.fft(alpha_imag.loc[i], norm = 'ortho').imag

for i in tqdm.tqdm(beta_real.index):
    beta_real.loc[i] = beta_real.loc[i] - beta_real.loc[i].mean()
    beta_imag.loc[i] = beta_imag.loc[i] - beta_imag.loc[i].mean()

    beta_real.loc[i] = np.fft.fft(beta_real.loc[i], norm = 'ortho').real
    beta_imag.loc[i] = np.fft.fft(beta_imag.loc[i], norm = 'ortho').imag

real_part = []
imag_part = []

for col in dst_list:
    real_part.append(col + '_fft_real')
    imag_part.append(col + '_fft_imag')

alpha_real.columns = real_part
alpha_imag.columns = imag_part
alpha = pd.concat((alpha_real, alpha_imag), axis = 1)

beta_real.columns = real_part
beta_imag.columns = imag_part
beta = pd.concat((beta_real, beta_imag), axis = 1)

x_train = pd.concat((x_train, alpha), axis = 1)
x_test = pd.concat((x_test, beta), axis = 1)

print(model_scoring_cv(estimator = multi_lgb,
                       x_data = x_train,
                       y_data = y_train))
'''
측정 빛에 이산 푸리에 변환을 적용
np.fft는 고속 푸리에 변환을 이용한 이산 푸리에 변환을 계산
실수부와 허수부로 나오는 결과물을 각각 변수로 저장
'''

## Remove src columns
x_train = x_train.drop(columns = src_list)
x_test = x_test.drop(columns = src_list)

print(model_scoring_cv(estimator = multi_lgb,
                       x_data = x_train,
                       y_data = y_train))

multi_lgb.fit(x_train, y_train)
pred = multi_lgb.predict(x_test)

pred = pd.DataFrame(data = pred,
                    columns = submission.columns,
                    index = submission.index)

pred.to_csv(path + '/comp1/submission_200625.csv')