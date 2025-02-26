# 107번을 카피해서 Activation을 넣고 튠하시오
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
from keras.activations import relu, elu, softmax

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
def build_model(drop, optimizer, learning_rate, activation): 
    inputs = Input(shape = (784,), name = 'input')
    x = Dense(512, activation = activation, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'output')(x)

    opt = optimizer(lr = learning_rate)     # optimizer와 learning_rate 엮어주기
    model = Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer = opt, metrics = ['accuracy'],
                  loss = 'categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [256, 128]
    optimizers = [Adam, Adadelta, Adamax, Nadam, RMSprop, Adagrad, SGD]
    activations = [relu, elu, softmax]
    dropout = np.linspace(0.1, 0.5, 5).tolist()  # 0.1 ~ 0.5까지 5단위로
    learning_rate = [0.1, 0.05, 0.25, 0.001]
    return {'batch_size': batches,
            'optimizer': optimizers,
            'drop': dropout,
            'activation': activations,
            'learning_rate': learning_rate}

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