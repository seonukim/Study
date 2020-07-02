# 100번을 카피해서 lr을 넣고 튠하시오
# LSTM -> Dense로 바꿀 것

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM
from keras.layers import Conv2D, Flatten
from keras.layers import MaxPooling2D, Dropout
from keras.optimizers import Adam, Adadelta, Adagrad
from keras.optimizers import RMSprop, Nadam, SGD, Adamax
from keras.wrappers.scikit_learn import KerasClassifier     # 케라스를 사이킷런으로 감싼다. (사이킷런에서 쓸 수 있게)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)

x_train = x_train.reshape(x_train.shape[0], 28*28) / 255            # 정규화(min_max)
x_test = x_test.reshape(x_test.shape[0], 28*28) / 255               # 정규화(min_max)
x_train = x_train.reshape(x_train.shape[0], 28*28) / 255            # 정규화(min_max)
x_test = x_test.reshape(x_test.shape[0], 28*28) / 255               # 정규화(min_max)
print(x_train.shape)                # (60000, 784)
print(x_test.shape)                 # (10000, 784)

y_train = np_utils.to_categorical(y_train)      # label이 0부터 시작함
y_test = np_utils.to_categorical(y_test)        # label이 0부터 시작함
print(y_train.shape)                # (60000, 10)
print(y_test.shape)                 # (10000, 10)


# 2. 모델링
def build_model(drop, optimizer, lr): 
    inputs = Input(shape = (784,), name = 'input')
    x = Dense(64, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(32, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(16, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer(lr = lr), metrics = ['accuracy'],
                  loss = 'categorical_crossentropy')
    return model

## 최적화 함수 정의
# adam = Adam(lr = lr)
# adadelta = Adadelta(lr = lr)
# adamax = Adamax(lr = lr)
# nadam = Nadam(lr = lr)
# rmsprop = RMSprop(lr = lr)
# adagrad = Adagrad(lr = lr)
# sgd = SGD(lr = lr)

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    optimizers = [Adam, Adadelta, Adamax, Nadam, RMSprop, Adagrad, SGD]
    dropout = np.linspace(0.1, 0.5, 5).tolist()
    lr = np.linspace(0.001, 0.01, 10).tolist()
    return {'batch_size': batches,
            'optimizer': optimizers,
            'drop': dropout,
            'lr': lr}

# KerasClassifier 모델 구성하기
model = KerasClassifier(build_fn = build_model, verbose = 1)

# hyperparameters 변수 정의
hyperparameters = create_hyperparameter()

search = RandomizedSearchCV(estimator = model,
                            param_distributions = hyperparameters, cv = 3)

# 모델 훈련
search.fit(x_train, y_train)
score = search.score(x_test, y_test)
print(search.best_params_)              # {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 20}
print("score : ", score)                # 0.9661999940872192