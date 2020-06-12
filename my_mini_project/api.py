import pandas as pd

## 데이터 로드
cpi = pd.read_csv('./my_mini_project/cpi.csv',
                  header = 0,
                  index_col = 0,
                  encoding = 'cp949')
print(cpi.shape)
print(cpi.head())

## 결측치 확인
print(cpi.isnull().sum())