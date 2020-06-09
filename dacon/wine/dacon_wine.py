import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
scaler = StandardScaler()
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10)

### 데이터 ###
train = pd.read_csv('./dacon/wine/train.csv',
                    index_col = 0,
                    header = 0)
test = pd.read_csv('./dacon/wine/test.csv',
                   index_col = 0,
                   header = 0)
print(train.shape)          # (5497, 13)
print(test.shape)           # (1000, 12)
print(train.head(n = 5))
print(test.head(n = 5))
'''
       quality  fixed acidity  volatile acidity  ...  sulphates  alcohol   type
index                                            ...
0            5            5.6             0.695  ...       0.44     10.2  white
1            5            8.8             0.610  ...       0.59      9.5    red
2            5            7.9             0.210  ...       0.52     10.9  white
3            6            7.0             0.210  ...       0.50     10.8  white
4            6            7.8             0.400  ...       0.43     10.9  white

       fixed acidity  volatile acidity  citric acid  ...  sulphates  alcohol   type      
index                                                ...
0                9.0              0.31         0.48  ...       0.38     11.6  white      
1               13.3              0.43         0.58  ...       0.49      9.0    red      
2                6.5              0.28         0.27  ...       0.69      9.4  white      
3                7.2              0.15         0.39  ...       0.47     10.0  white      
4                6.8              0.26         0.26  ...       0.47     11.8  white 
'''

## 결측치 확인 ##
print(train.isnull().sum())     # 확인
print(test.isnull().sum())      # 확인

## 시각화 ##
quality_count = train.groupby('quality')['quality'].count()
# quality_count.plot()
# plt.show()

## train 데이터 나누기 ##
x = train.iloc[:, 1:]
y = train.iloc[:, :1]
print(x.shape)          # (5497, 12)      
print(y.shape)          # (5497, 1)

## numpy 형 변환 ##
x = x.values
y = y.values
print(type(x))          # <class 'numpy.ndarray'>
print(type(y))          # <class 'numpy.ndarray'>

## 데이터 나누기 ##
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = 0.2, random_state = 77)
print(x_train.shape)    # (4397, 12)
print(x_test.shape)     # (1100, 12)
print(y_train.shape)    # (4397, 1)
print(y_test.shape)     # (1100, 1)

## 정규화 ##
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(x_train[0])
# print(x_test[0])

## 원핫인코딩 ##
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)

### 모델링 ###
