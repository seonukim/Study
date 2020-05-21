from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 구성
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = array([4, 5, 6, 7])
# y2 = array([[4, 5, 6, 7]])            # res : (1, 4)
# y3 = array([[4], [5], [6], [7]])      # res : (4, 1)

print("x.shape : ", x.shape)        # res : (4, 3)
print("y.shape : ", y.shape)        # res : (4, )

# x = x.reshape(4, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
#                 4           3       1
print(x.shape)
'''reshape 후 검산을 해야함 -> 모두 곱해서 reshape 전후가 같은 값이 나오면 문제 없음'''


# 2. 모델 구성
model = Sequential()
# model.add(LSTM(10, activation = 'relu', input_shape = (3, 1)))      # column의 갯수와 몇개씩 자를 것인지
model.add(LSTM(5, input_length = 3, input_dim = 1))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

'''
# 3. 실행
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y, epochs = 100, batch_size = 1)

x_input = array([5, 6, 7])
x_input = x_input.reshape(1, 3, 1)

print(x_input)

y_hat = model.predict(x_input)
print(y_hat)
'''