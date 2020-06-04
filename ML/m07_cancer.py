import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from keras.utils import np_utils
ss = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rs = RobustScaler()
pca = PCA(n_components = 25)


#### 분류 모델
### 1. 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)                  # (569, 30)
print(y.shape)                  # (569,)

## 1-1. 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)            # (455, 30)
print(x_test.shape)             # (114, 30)
print(y_train.shape)            # (455,)
print(y_test.shape)             # (114,)

## 1-2. PCA
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)            # (455, 25)
print(x_test.shape)             # (114, 25)

## 1-3. Scaler
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)
print(x_train[0])               # [0.10144103 0.48618328 0.13054923 0.33266537 0.47898069]
print(x_test[0])                # [0.08751287 0.45397927 0.14563761 0.30956911 0.29970822]


### 2. 모델링
rfc = RandomForestClassifier(n_estimators = 100, max_depth = 20)

model = rfc


### 3. 모델 훈련
model.fit(x_train, y_train)


### 4. 모델 평가 및 결과 예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("y의 예측값 : \n", y_pred[:5])
print("모델 정확도 : ", acc)

### 5. 결과
'''
y의 예측값 :
 [1 0 1 0 0]

모델 정확도 :  0.9473684210526315
'''

'''
#### 회귀모델
### 1. 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)                  # (569, 30)
print(y.shape)                  # (569,)

## 1-1. 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)            # (455, 30)
print(x_test.shape)             # (114, 30)
print(y_train.shape)            # (455,)
print(y_test.shape)             # (114,)

## 1-2. PCA
# pca.fit(x_train)
# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)
# print(x_train.shape)            # (455, 5)
# print(x_test.shape)             # (114, 5)

## 1-3. Scaler
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)
print(x_train[0])               # [0.10144103 0.48618328 0.13054923 0.33266537 0.47898069]
print(x_test[0])                # [0.08751287 0.45397927 0.14563761 0.30956911 0.29970822]


### 2. 모델링
rfc = RandomForestRegressor(n_estimators = 100, max_depth = 20)

model = rfc


### 3. 모델 훈련
model.fit(x_train, y_train)


### 4. 모델 평가 및 결과 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("y의 예측값 : \n", y_pred[:5])
print("모델 정확도 : ", r2)
'''

### 5. 결과
'''
y의 예측값 :
 [1. 0. 1. 0. 0.]

모델 정확도 :  0.8192875897435897
'''