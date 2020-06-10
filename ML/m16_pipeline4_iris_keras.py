# iris를 케라스 파이프라인으로 구성
# 당연히 RandomizedSearchCV 구성

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU, Input
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import load_iris
from keras.optimizers import SGD
sgd = SGD()
scaler = StandardScaler()
pca = PCA(n_components = 3)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)
leaky = LeakyReLU(alpha = 0.3)

### 1. 데이터
x, y = load_iris(return_X_y = True)
print(x.shape)          # (150, 4)
print(y.shape)          # (150,)

## 1-1. 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2)
print(x_train.shape)        # (120, 4)
print(x_test.shape)         # (30, 4)
print(y_train.shape)        # (120,)
print(y_test.shape)         # (30,)

## 1-2. 레이블 인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)        # (120, 3)
print(y_test.shape)         # (30, 3)

## 1-3. PCA
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)        # (120, 3)
print(x_test.shape)         # (30, 3)


### 2. 모델링
def mymodel(drop = 0.2, optimizer = 'adam'):
    input1 = Input(shape = (3, ))
    x = Dense(16, activation = leaky)(input1)
    x = Dense(14, activation = leaky)(x)
    x = Dropout(rate = drop)(x)
    x = Dense(12, activation = leaky)(x)
    x = Dense(10, activation = leaky)(x)
    x = Dropout(rate = drop)(x)
    output = Dense(3, activation = 'softmax')(x)
    model = Model(inputs = input1, outputs = output)
    model.compile(loss = 'categorical_crossentropy',
                  metrics = ['accuracy'],
                  optimizer = optimizer)
    return model

def hyperParams():
    batch_size = [1, 10, 20, 32]
    epochs = [20, 40, 60, 80, 100]
    # dropout = np.linspace(0.1, 0.5, 5)
    return {'model__batch_size': batch_size,
            'model__epochs': epochs}

## 2-1. keras 모델 구성
model = KerasClassifier(build_fn = mymodel, verbose = 1)

## 2-2. 파라미터 변수 생성
params = hyperParams()

## 2-3. 파이프라인 정의
pipe = Pipeline([('scaler', StandardScaler()),('model', model)])

## 2-4. RandomSearchCV 정의
search = RandomizedSearchCV(estimator = pipe,
                            param_distributions = params,
                            cv = 3)


### 3. 모델 훈련
search.fit(x_train, y_train)


### 4. 모델 평가 및 예측
acc = search.score(x_test, y_test)
print("acc : ", acc)

print("최적의 매개변수 : ", search.best_params_)
print("최적의 매개변수 : ", search.best_estimator_)