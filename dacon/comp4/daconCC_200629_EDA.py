import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings ; warnings.filterwarnings('ignore')
path = 'C:/Users/bitcamp/Desktop/dacon/'
os.chdir(path)

## data
train = pd.read_csv(path + '201901-202003.csv')
# print(train.isnull().sum())
train = train.fillna('')
# print(train.isnull().sum())
train = train[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
train = train.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop = False)
print("=" * 40)
print(train.head())

submit = pd.read_csv(path + 'submission.csv', index_col = 0)
submit = submit.loc[submit['REG_YYMM'] == 202004]           # 2020년 4월 데이터만 submit에 저장
submit = submit[['CARD_SIDO_NM', 'STD_CLSS_NM']]            # 2020년 4월의 카드이용지역, 업종명만 가져옴
print("=" * 40)
print(submit.head())
print(submit.shape)     # (697, 2)

REG_YYMMs = np.sort(train['REG_YYMM'].unique())             # unique() : 유일한 값 찾기, 같은 값 제거
print("=" * 40)
print(REG_YYMMs.shape)          # (15,) ; train데이터는 201901 ~ 202003까지의 데이터, REG_YYMM의 동일연월을 제거함

AMTs = []
for REG_YYMM in REG_YYMMs:
    df = train.loc[train['REG_YYMM'] == REG_YYMM]
    df = df[['CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
    tmp = submit.merge(df, left_on = ['CARD_SIDO_NM', 'STD_CLSS_NM'],
                           right_on = ['CARD_SIDO_NM', 'STD_CLSS_NM'], how = 'left')
    tmp = tmp.fillna(0)
    AMT = list(tmp['AMT'])
    AMTs.append(AMT)
AMTs = np.array(AMTs)
print("=" * 40)
print(AMTs.shape)       # (15, 697)

def plot_graph(dt, AMTs):
    for i in range(dt, len(AMTs)):
        tmp1 = np.log1p(AMTs[i-dt])
        tmp2 = np.log1p(AMTs[i])
        corr = np.corrcoef(tmp1, tmp2)[0][1]
        plt.title('Correlation: %.4f'%(corr))
        plt.scatter(tmp1, tmp2, color = 'k', alpha = 0.1)
        plt.xlabel('Log AMT, %s'%(REG_YYMMs[i-dt]))
        plt.ylabel('Log AMT, %s'%(REG_YYMMs[i]))
        plt.xlim(tmp1.min(), tmp1.max())
        plt.ylim(tmp1.min(), tmp2.max())
        plt.show()

plot_graph(1, AMTs)
# plot_graph(3, AMTs)
# plot_graph(6, AMTs)
