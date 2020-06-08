import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
scaler = MinMaxScaler()
# scaler = StandardScaler()
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 데이터
train = pd.read_csv('./data/dacon/comp1/train.csv',
                    header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv',
                   header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv',
                         header = 0, index_col = 0)

print('train.shape : ', train.shape)                # (10000, 75) : x_train, test
print('test.shape : ', test.shape)                  # (10000, 71) : x_predict
print('submission.shape : ', submission.shape)      # (10000, 4)  : y_predict

# Null 확인
print("=" * 40)
print(train.isnull().sum())

# 보간법 - 선형 보간법
train = train.interpolate()         # predict 값으로 채워넣어주는 값이다
print("=" * 40)
print(train.isnull().sum())

test = test.interpolate()
print("=" * 40)
print(test.isnull().sum())

# Nan 확인
print("=" * 40)
print(train.isna().sum())

print("=" * 40)
print(test.isna().sum())

test = test.fillna(method = 'bfill')
print("=" * 40)
print(test.isna().sum())


# 서브미션 파일을 만든다
# y_pred.to_csv(경로)


x = train.iloc[:, :71]
y = train.iloc[:, 71:]
print(x.shape)          # (10000, 71)
print(y.shape)          # (10000, 4)

np.save('./dacon/x_data', arr = x)
np.save('./dacon/y_data', arr = y)

# NumPy 데이터 로드
x = np.load('./dacon/x_data.npy')
y = np.load('./dacon/y_data.npy')
print(x.shape)          # (10000, 71)
print(y.shape)          # (10000, 4)

# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.25,
    shuffle = True, random_state = 77)
print(x_train.shape)        # (7500, 71)
print(x_test.shape)         # (2500, 71)
print(y_train.shape)        # (7500, 4)
print(y_test.shape)         # (2500, 4)

# 정규화
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])
print(x_test[0])

# 모델 구성
model = Sequential()

model.add(Dense(8, input_shape = (71, ),
                activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(4, activation = 'relu'))

model.summary()



# 컴파일 및 훈련
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 50, batch_size = 32,
                 validation_split = 0.2, verbose = 1)
                 

# 모델 평가 및 예측
res = model.evaluate(x_test, y_test)
print("loss : ", res[0])
print("mae : ", res[1])


predict = model.predict(x_test)
print("predict : \n", predict[:5])


