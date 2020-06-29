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
train = train.fillna('')
df = train.copy()
df = df[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
df = df.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop = False)
df = df.loc[df['REG_YYMM'] == 202003]
df = df[['CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]

submit = pd.read_csv(path + 'submission.csv', index_col = 0)
submit = submit.loc[submit['REG_YYMM'] == 202004]
submit = submit[['CARD_SIDO_NM', 'STD_CLSS_NM']]
submit = submit.merge(df, left_on = ['CARD_SIDO_NM', 'STD_CLSS_NM'],
                          right_on = ['CARD_SIDO_NM', 'STD_CLSS_NM'], how = 'left')
submit = submit.fillna(0)
AMT = list(submit['AMT']) * 2

submit = pd.read_csv(path + 'submission.csv', index_col = 0)
submit['AMT'] = AMT
submit.to_csv('submission.csv', encoding = 'utf-8-sig')
print(submit.head())