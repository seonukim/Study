import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'

es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     mode = 'auto', save_best_only = True)
scaler = RobustScaler()
pca = PCA(n_components = 10)


# 1. 데이터
x, y = load_boston(return_X_y = True)
# print(x.shape)
# print(y.shape)

# 1-1. 데이터 분할


# 1-2. Scaling
pca.fit(x)
x = pca.transform(x)
# x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True,
    random_state = 77)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# 2. 모델링
model = Sequential()
model.add(Dense(32, input_shape = (10, ), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

model.summary()


# 3. 컴파일 및 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x_train, y_train, callbacks = [es, cp],
          epochs = 100, batch_size = 32, verbose = 1,
          validation_split = 0.25)              # key_error : 'val_loss', cp에 val_loss를 선언했는데 validation셋을 안줬기 때문

# print(hist.history.keys())


# 4. 모델 평가
res = model.evaluate(x_test, y_test)
print(res)

y_pred = model.predict(x_test)
print("y_pred : ", y_pred)

# 5. 성능 지표 평가
print("R2 : ", r2_score(y_test, y_pred))


'''
Result
R2 Score = 0.7801983681184715
'''