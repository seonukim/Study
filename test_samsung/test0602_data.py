# 2020.06.02 시험 리뷰

import numpy as np
import pandas as pd

samsung = pd.read_csv('./data/csv/samsung.csv',
                      index_col = 0,
                      header = 0,
                      sep = ',',
                      encoding = 'cp949')

hite = pd.read_csv('./data/csv/hite.csv',
                      index_col = 0,
                      header = 0,
                      sep = ',',
                      encoding = 'cp949')

print(samsung)
print(hite)
print(samsung.shape)                # (700, 1)
print(hite.shape)                   # (720, 5)


# None 제거1
samsung = samsung.dropna(axis = 0)
print(samsung)
print(samsung.shape)                # (509, 1)
hite = hite.fillna(method = 'bfill')
hite = hite.dropna(axis = 0)
print(hite)
print(hite.shape)                   # (509, 5)


# None 제거2
hite = hite[0:509]
hite.iloc[0, 1:5] = [10, 20, 30, 40]
hite.loc['2020-06-02, '고가':'거래량'] = ['10', '20', '30', '40']
print(hite)