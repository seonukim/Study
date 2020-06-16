import pandas as pd

## 데이터
train = pd.read_csv('./my_mini_project/cpi_train.csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
test = pd.read_csv('./my_mini_project/cpi_test.csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
print(train.shape)      # (461, 47)
print(test.shape)       # (461, 46)
