import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

## 데이터
train = pd.read_csv('./my_mini_project/cpi_train.csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
test = pd.read_csv('./my_mini_project/cpi_test.csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
print(train.shape)          # (461, 47)
print(test.shape)           # (461, 46)

## 데이터 나누기
x = train.iloc[:, :46]
y = train.iloc[:, 46:]
print(x.shape)              # (461, 46)
print(y.shape)              # (461, 1)

## train, test로 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = False)
print(x_train.shape)        # (368, 46)
print(x_test.shape)         # (93, 46)
print(y_train.shape)        # (368, 1)
print(y_test.shape)         # (93, 1)

## scaling
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])