from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
sc = MinMaxScaler()
scaler = StandardScaler()

# 1. 데이터 구성
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11], [10, 11, 12],
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_predict = array([50, 60, 70])

print("x.shape : ", x.shape)        # res : (13, 3)
print("y.shape : ", y.shape)        # res : (13, )

x = x.reshape(x.shape[0], x.shape[1], 1)
#                13           3       1

# x = sc.fit_transform(x)
# y = sc.fit_transform(y)

# print(x)
# print(y)

'''
                행          열          몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
batch_size = 행을 기준으로 자름
feature = 원소 하나하나 자름

input_shape = (timesteps, feature)
input_length = timesteps, input_dim = feature
'''

# print(x.shape)
'''reshape 후 검산을 해야함 -> 모두 곱해서 reshape 전후가 같은 값이 나오면 문제 없음'''

# 1-2.  데이터 분할
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size = 0.2, shuffle = True,
#     random_state = 1234)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)




# 2. 모델 구성
model = Sequential()
model.add(LSTM(101, activation = 'tanh', input_length = 3, input_dim = 1, use_bas = False))
model.add(Dense(800))
model.add(Dense(300))
model.add(Dense(456))
model.add(Dense(504))
# model.add(Dense(80))
# model.add(Dense(60))
# model.add(Dense(40))
# model.add(Dense(80))
# model.add(Dense(30))
# model.add(Dense(50))
# model.add(Dense(80))
# model.add(Dense(30))
model.add(Dense(1))

model.summary()


# 3. 실행
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y,
          epochs = 10000, batch_size = 2,
          callbacks = [early])

x_predict = x_predict.reshape(1, 3, 1)


# 4. 평가 및 예측
# res = model.evaluate(x_test, y_test, batch_size = 1)
# print(res)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
