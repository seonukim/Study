import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
kf = KFold(n_splits = 5)
scaler = StandardScaler()


## 데이터 불러오기
train = pd.read_csv('./my_mini_project/cpi_train(1975.01 - 2002.09).csv',
                    index_col = 0, header = 0, encoding = 'cp949')
test = pd.read_csv('./my_mini_project/cpi_test(2002.10 - 2020.05).csv',
                   index_col = 0, header = 0, encoding = 'cp949')
print(train.shape)      # (213, 13)
print(test.shape)       # (213, 13)

## 데이터 분할하기
def split_xy(data, time, y_column):
    x, y = list(), list()
    for i in range(len(data)):
        x_end_number = i + time
        y_end_number = x_end_number + y_column

        if y_end_number > len(data):
            break
        tmp_x = data.iloc[i:x_end_number, :]
        tmp_y = data.iloc[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return pd.DataFrame(x), pd.DataFrame(y)

