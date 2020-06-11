import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action = 'ignore')

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor


## 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv',
                    index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv',
                   index_col = 0, header = 0)
submit = pd.read_csv('./data/dacon/comp1/sample_submission.csv',
                     index_col = 0, header = 0)
print("train_data : \n", train.head())
print("test_data : \n",test.head())
print("submit_data : \n",submit.head())
'''
train_data :
     rho  650_src  660_src  670_src  680_src  ...       990_dst    hhb  hbo2     ca    na     
id                                           ...
0    25  0.37950  0.42993  0.52076  0.57166  ...  4.378513e-17   5.59  4.32   8.92  4.29      
1    10  0.00000  0.00000  0.01813  0.00000  ...           NaN   0.00  2.83   7.25  4.64      
2    25  0.00000  0.03289  0.02416  0.03610  ...           NaN  10.64  3.00   8.40  5.16      
3    10  0.27503  0.31281  0.32898  0.41041  ...           NaN   5.67  4.01   5.05  4.35      
4    15  1.01521  1.00872  0.98930  0.98874  ...           NaN  11.97  4.41  10.78  2.42      

[5 rows x 75 columns]

test_data :
        rho  650_src  660_src  670_src  ...       960_dst       970_dst       980_dst       990_dst
id                                     ...

10000   15  0.15406  0.23275  0.30977  ...  1.429966e-14  0.000000e+00           NaN  7.320236e-14
10001   15  0.48552  0.56939  0.67575  ...  2.282485e-14  7.348414e-14  1.259055e-13  2.349874e-13
10002   10  0.46883  0.56085  0.62442  ...           NaN  1.219010e-11           NaN
 NaN
10003   10  0.06905  0.07517  0.10226  ...  0.000000e+00  3.304247e-12  4.106134e-11
 NaN
10004   25  0.00253  0.00757  0.01649  ...  0.000000e+00  0.000000e+00  1.910775e-16  2.215673e-15

[5 rows x 71 columns]

submit_data :
        hhb  hbo2  ca  na
id
10000    0     0   0   0
10001    0     0   0   0
10002    0     0   0   0
10003    0     0   0   0
10004    0     0   0   0
'''

## 결측치 확인
print("=" * 40)
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

## 결측치 처리
train = train.fillna(train.mean())
test = test.fillna(test.mean())
print("=" * 40)
print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])

## train, test 데이터 분리
x_train = train.iloc[:, :71]
y_train = train.iloc[:, 71:]
print("=" * 40)
print(x_train.shape)
print(y_train.shape)

## Multioutput Layer
model = MultiOutputRegressor(XGBRegressor(learning_rate = 0.1,
                                          max_depth = 10,
                                          n_estimators = 100,
                                          n_jobs = -1))

model.fit(x_train, y_train)

y_pred = model.predict(test)
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('./dacon/my_submission.csv')