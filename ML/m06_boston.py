import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras.utils import np_utils
ss = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rs = RobustScaler()
pca = PCA(n_components = 10)


### 회귀 모델
### 1. 데이터
x, y = load_boston(return_X_y = True)
print(x.shape)                  # (506, 13)
print(y.shape)                  # (506,)

## 1-1. 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)            # (404, 13)
print(x_test.shape)             # (102, 13)
print(y_train.shape)            # (404,)
print(y_test.shape)             # (102,)

## 1-2. Scaler
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)
print(x_train[0])               # [0.16116697 0.30235392 0.4347566 ]
print(x_test[0])                # [0.2030123  0.29580018 0.15580581]

## 1-3. PCA
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)            # (404, 10)
print(x_test.shape)             # (102, 10)

### 2. 모델링
model = RandomForestRegressor(n_estimators = 100, max_depth = 20)
# model = RandomForestClassifier(n_estimators = 100, max_depth = 20)
# model = SVC()
# model = LinearSVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = SVR()


### 3. 모델 훈련
model.fit(x_train, y_train)


### 4. 모델 평가 및 결과 예측
# x_test = mms.inverse_transform(x_test)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
score = model.score(x_train, y_train)
mse = mean_squared_error(y_test, y_pred)
print("y의 예측값 : \n", y_pred[:5])
print("모델 정확도 : ", r2)
print("모델 score : ", score)
print("mse : ", mse)


### 5. 결과
'''
y의 예측값 :
 [22.384 17.618 22.756 34.483 23.627]

모델 정확도 :  0.7825015468828667
모델 score :  0.9683398519440181
'''



#### 분류 모델
### 1. 데이터
x, y = load_boston(return_X_y = True)
print(x.shape)                  # (506, 13)
print(y.shape)                  # (506,)

## 1-1. 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)            # (404, 13)
print(x_test.shape)             # (102, 13)
print(y_train.shape)            # (404,)
print(y_test.shape)             # (102,)

## 1-2. 원핫인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train[0])               # [0. 1. 0.]
print(y_test[0])                # [0. 1. 0.]


## 1-4. Scaler
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)
print(x_train[0])               # [0.16116697 0.30235392 0.4347566 ]
print(x_test[0])                # [0.2030123  0.29580018 0.15580581]

## 1-3. PCA
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)            # (404, 5)
print(x_test.shape)             # (102, 5)

### 2. 모델링
# model = RandomForestRegressor(n_estimators = 100, max_depth = 20)
model = RandomForestClassifier(n_estimators = 100, max_depth = 20)
# model = SVC()
# model = LinearSVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = SVR()


### 3. 모델 훈련
model.fit(x_train, y_train)


### 4. 모델 평가 및 결과 예측
# x_test = mms.inverse_transform(x_test)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
score = model.score(x_train, y_train)
print("y의 예측값 : \n", y_pred[:5])
print("모델 정확도 : ", acc)
print("모델 score : ", score)


### 5. 결과
'''
y의 예측값 :
 [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0.]]
  
모델 정확도 :  0.0196078431372549
모델 score :  1.0
'''