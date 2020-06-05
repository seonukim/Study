import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils

ss = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rs = RobustScaler()
# pca = PCA(n_components = 10)

### 1. 데이터
wine = pd.read_csv('./data/csv/winequality-white.csv',
                   header = 0, index_col = None,
                   sep = ';', encoding = 'cp949')
print(wine.head())
print(wine.tail())
print(wine.shape)                   # (4898, 12)

## 1-1. 데이터 전처리
# 1-1-1. 결측치 확인
print(wine.isna())                  # 확인 ok

## 1-2. numpy 파일로 변환 후 저장
wine = wine.values
print(type(wine))                   # <class 'numpy.ndarray'>
print(wine)
print(wine.shape)                   # (4898, 12)
np.save('./data/wine_np.npy', arr = wine)

## 1-3. numpy 파일 불러오기
np.load('./data/wine_np.npy')
print(wine.shape)                   # (4898, 12)

## 1-4. 데이터 나누기
x = wine[:, 0:11]
y = wine[:, -1:]
print(x.shape)                      # (4898, 11)
print(y.shape)                      # (4898, 1)

## 1-5. train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2)
print(x_train.shape)                # (4408, 11)
print(x_test.shape)                 # (490, 11)
print(y_train.shape)                # (4408, 1)
print(y_test.shape)                 # (490, 1)

## 1-6. 원핫인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                # (4408, 10)
print(y_test.shape)                 # (490, 10)


## 1-7. 데이터 Scaling
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)
print(x_train[0])                   # [0.33653846 0.21621622 0.25903614 0.01687117 0.24315068 0.12543554
                                    #  0.31888112 0.06499518 0.41666667 0.23255814 0.77419355]
print(x_test[1])                    # [0.40384615 0.10810811 0.29518072 0.01840491 0.17808219 0.04878049
                                    #  0.38041958 0.13635487 0.4537037  0.30232558 0.32258065]

## 1-8. PCA
# pca.fit(x_train)
# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)
# print(x_train.shape)                # (4408, 10)
# print(x_test.shape)                 # (490, 10)


# 2. 모델링
model = RandomForestClassifier(n_estimators = 500, max_depth = 6)


# 3. 모델 훈련
model.fit(x_train, y_train)


# 4. 모델 평가
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("분류 정확도 : ", acc)                # 0.5420408163265306
