import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras.layers import Conv1D, Flatten, MaxPooling1D, Input
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()

leaky = LeakyReLU(alpha = 0.2)
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10)

### 데이터 ###
x = pd.read_csv('./data/dacon/comp3/train_features.csv',
                encoding = 'utf-8')
y = pd.read_csv('./data/dacon/comp3/train_target.csv',
                index_col = 0, header = 0,
                encoding = 'utf-8')
x_pred = pd.read_csv('./data/dacon/comp3/test_features.csv',
                     encoding = 'utf-8')
print(x.shape)                # (1050000, 6)
print(y.shape)                # (2800, 5)
print(x_pred.shape)           # (262500, 6)

x_train = x
y_train = y
print(x_train.shape)        # (1050000, 6)
print(y_train.shape)        # (2800, 5)
print(x_train.head())


x_train = x_train.drop('Time', axis = 1)
print(x_train.head())


x_train = np.sqrt(x_train.groupby(x_train['id']).mean())
print(x_train.shape)        # (2800, 4)


x_train = pd.read_csv('./data/dacon/comp3/x_Train.csv',
                      index_col = 0, header = 0,
                      encoding = 'utf-8')
print(x_train.head())
print(x_train.shape)        # (2800, 4)

print(y_train.head())
print(y_train.shape)        # (2800, 4)


x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size = 0.2)
print(x_train.shape)        # (2240, 4)
print(x_test.shape)         # (560, 4)
print(y_train.shape)        # (2240, 4)
print(y_test.shape)         # (560, 4)

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 4, 1)
x_test = x_test.reshape(-1, 4, 1)
print(x_train.shape)        # (2240, 4, 1)
print(x_test.shape)         # (560, 4, 1)


# 2. 모델링
def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (4, 1), name = 'input')
    x1 = Conv1D(filters = 64, kernel_size = 3,
                padding = 'same', activation = leaky)(inputs)
    x1 = MaxPooling1D()(x1)
    x1 = Dropout(drop)(x1)
    x1 = Conv1D(filters = 32, kernel_size = 3,
                padding = 'same', activation = leaky)(x1)
    x1 = MaxPooling1D()(x1)
    x1 = Dropout(drop)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(16, activation = leaky, name = 'hidden3')(x1)
    x1 = Dropout(drop)(x1)
    outputs = Dense(4, activation = leaky, name = 'output')(x1)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['mse'],
                  loss = 'mse')
    return model

build_model().summary()

def create_hyperparameter():
    batches = [50, 100, 150, 200, 250, 300]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return {'batch_size': batches,
            'optimizer': optimizers,
            'drop': dropout}

# KerasRegressor 모델 구성하기
model = KerasRegressor(build_fn = build_model, verbose = 1)

# hyperparameters 변수 정의
hyperparameters = create_hyperparameter()

search = RandomizedSearchCV(estimator = model,
                            param_distributions = hyperparameters, cv = 5)

# 훈련
search.fit(x_train, y_train)
'''
x_pred = x_pred.drop('Time', axis = 1)
# print(x_pred.head())

x_pred = np.sqrt(x_pred.groupby(x_pred['id']).mean())
# print(x_pred.shape)        # (700, 4)

x_pred = x_pred.values
x_pred = scaler.fit_transform(x_pred)
x_pred = x_pred.reshape(-1, 4, 1)
# print(type(x_pred))         # <class 'numpy.ndarray'>

y_predict = model.predict(x_pred)
print("Predict : \n", y_predict)


submit = pd.DataFrame(y_predict)
print(submit.head())

submit.to_csv('./data/dacon/comp3/mysubmit_2.csv')
'''