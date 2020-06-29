import os
import time
import warnings ; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = 'C:/Users/bitcamp/Desktop/dacon/'
os.chdir(path)

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor


### 데이터 전처리
## Data Cleansing & Pre-Processing
def grap_year(data):
    data = str(data)
    return int(data[:4])

def grap_month(data):
    data = str(data)
    return int(data[4:])

## 날짜 처리
data = pd.read_csv(path + '201901-202003.csv')
data = data.fillna('')
data['year'] = data['REG_YYMM'].apply(lambda x: grap_year(x))
data['month'] = data['REG_YYMM'].apply(lambda x: grap_month(x))
data = data.drop(['REG_YYMM'], axis = 1)

## 데이터 정제
df = data.copy()
df = df.drop(['CARD_CCG_NM', 'HOM_CCG_NM'], axis = 1)

columns = ['CARD_SIDO_NM', 'STD_CLSS_NM', 'HOM_SIDO_NM', 'AGE', 'SEX_CTGO_CD', 'FLC', 'year', 'month']
df = df.groupby(columns).sum().reset_index(drop = False)

## 인코딩
dtypes = df.dtypes
encoders = {}
for column in df.columns:
    if str(dtypes[column]) == 'object':
        encoder = LabelEncoder()
        encoder.fit(df[column])
        encoders[column] = encoder
    
df_num = df.copy()
for column in encoders.keys():
    encoder = encoders[column]
    df_num[column] = encoder.transform(df[column])

### 변수 선택 및 모델 구축
## feature, target 설정
train_num = df_num.sample(frac = 1, random_state = 0)
train_features = train_num.drop(['CSTMR_CNT', 'AMT', 'CNT'], axis = 1)
train_target = np.log1p(train_num['AMT'])

### 모델 학습 및 검증
## Model Tuning & Evaluation
# 훈련
model = XGBRegressor(n_estimators = 150,
                     learning_rate = 0.1,
                     max_depth = 20,
                     objective = 'reg:linear',
                     colsample_bytree = 0.7,
                     colsample_bylevel = 0.7,
                     importance_type = 'gain',
                     random_state = 18,
                     n_jobs = -1)
model.fit(train_features, train_target)

### 결과 및 결언
## Conclusion & Discussion
# 예측 템플릿 만들기
CARD_SIDO_NMs = df_num['CARD_SIDO_NM'].unique()
STD_CLSS_NMs = df_num['STD_CLSS_NM'].unique()
HOM_SIDO_NMs = df_num['HOM_SIDO_NM'].unique()
AGEs = df_num['AGE'].unique()
SEX_CTGO_CDs = df_num['SEX_CTGO_CD'].unique()
FLCs = df_num['FLC'].unique()
years = [2020]
months = [4, 7]

tmp = []
for CARD_SIDO_NM in CARD_SIDO_NMs:
    for STD_CLSS_NM in STD_CLSS_NMs:
        for HOM_SIDO_NM in HOM_SIDO_NMs:
            for AGE in AGEs:
                for SEX_CTGO_CD in SEX_CTGO_CDs:
                    for FLC in FLCs:
                        for year in years:
                            for month in months:
                                tmp.append([CARD_SIDO_NM, STD_CLSS_NM, HOM_SIDO_NM, AGE, SEX_CTGO_CD, FLC, year, month])
tmp = np.array(tmp)
tmp = pd.DataFrame(data = tmp,
                   columns = train_features.columns)

# 예측
pred = model.predict(tmp)
pred = np.expm1(pred)
tmp['AMT'] = np.round(pred, 0)
tmp['REG_YYMM'] = tmp['year'] * 100 + tmp['month']
tmp = tmp[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
tmp = tmp.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop = False)

# 디코딩
tmp['CARD_SIDO_NM'] = encoders['CARD_SIDO_NM'].inverse_transform(tmp['CARD_SIDO_NM'])
tmp['STD_CLSS_NM'] = encoders['STD_CLSS_NM'].inverse_transform(tmp['STD_CLSS_NM'])

# 제출 파일 만들기
submission = pd.read_csv(path + 'submission.csv', index_col = 0)
submission = submission.drop(['AMT'], axis = 1)
submission = submission.merge(tmp, left_on = ['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'],
                                   right_on = ['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], how = 'left')
submission.index.name = 'id'
submission.to_csv(path + 'mysubmission_200629(2).csv', encoding = 'utf-8-sig')
print(submission.head())