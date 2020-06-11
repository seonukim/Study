import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from xgboost import XGBRFClassifier
# scaler = StandardScaler()
scaler = MinMaxScaler()
pca = PCA(n_components = 4)
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10)
lr = LeakyReLU(alpha = 0.2)
ohe = OneHotEncoder()

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
       quality  fixed acidity  volatile acidity  citric acid  ...    pH  sulphates  alcohol   type
index                                                         ...
0            5            5.6             0.695         0.06  ...  3.44       0.44     10.2  white
1            5            8.8             0.610         0.14  ...  3.19       0.59      9.5    red
2            5            7.9             0.210         0.39  ...  3.05       0.52     10.9  white
3            6            7.0             0.210         0.31  ...  3.26       0.50     10.8  white
4            6            7.8             0.400         0.26  ...  3.04       0.43     10.9  white

[5 rows x 13 columns]

       fixed acidity  volatile acidity  citric acid  residual sugar  ...    pH  sulphates  alcohol   type      
index                                                                ...
0                9.0              0.31         0.48             6.6  ...  2.90       0.38     11.6  white      
1               13.3              0.43         0.58             1.9  ...  3.06       0.49      9.0    red      
2                6.5              0.28         0.27             5.2  ...  3.19       0.69      9.4  white      
3                7.2              0.15         0.39             1.8  ...  3.52       0.47     10.0  white      
4                6.8              0.26         0.26             2.0  ...  3.16       0.47     11.8  white 
[5 rows x 12 columns] 
'''

## type 컬럼 바꿔주기
train['type'] = train['type'].map({'white':0, 'red':1}).astype('int64')
test['type'] = test['type'].map({'white':0, 'red':1}).astype('int64')

x = train.drop('quality', axis = 1)
y = train['quality'].astype('int64')
print(x.head())
print(y.head())

# y 레이블 축소
# newlist = []
# for i in list(y):
#     if i <= 3:
#         newlist += [0]
#     elif i <= 5:
#         newlist += [1]
#     elif i <= 7:
#         newlist += [2]
#     elif i <= 9:
#         newlist += [3]
#     else:
#         newlist += [4]
# y = newlist

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape)

### 모델링 ###
# model = Sequential()
# model.add(Dense(16, input_shape = (12, ),
#                 activation = lr))
# model.add(Dropout(rate = 0.2))
# model.add(Dense(7, activation = 'softmax'))

# model.summary()

# ### 컴파일 및 훈련 ###
# model.compile(loss = 'categorical_crossentropy',
#               optimizer = 'adam', metrics = ['acc'])
# model.fit(x_train, y_train, epochs = 50, callbacks = [es],
#           batch_size = 10, validation_split = 0.2)
       
# ## 평가 및 예측 ###
# res = model.evaluate(x_test, y_test)
# print("loss : ", res[0])
# print("Acc : ", res[1])

### 모델
model = RandomForestClassifier(n_estimators = 300,
                               n_jobs = -1)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

y_pred = model.predict(test)
y_pred = ohe.inverse_transform(y_pred)
y_pred = np.argmax(y_pred, axis = 1)
print("Predict : \n", y_pred)

y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('./dacon/wine/y_pred_1.csv')
