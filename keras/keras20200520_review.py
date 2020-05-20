'''LSTM(장단기 메모리; Long Short-Term Memory)'''

# 1. 모듈 임포트
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

# 2. 데이터 구성
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6, 7])

print(x.shape)      # res : (4, 3)
print(y.shape)      # res : (4, )
'''y의 차원이 (4, )인 이유: y가 Scala 4개로 이루어진 벡터 한개이기 때문이다'''

# 2-1. 데이터 reshape
x = x.reshape(x.shape[0], x.shape[1], 1)

print(x.shape)


# 3. 모델 구성
model = Sequential()
#  model.add(LSTM(10, activation = 'relu', input_shape = (3, 1)))
model.add(LSTM(10, activation = 'relu', input_dim = 1, input_length = 3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


'''
# 4. 모델 학습
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y, epochs = 100, batch_size = 1)


# 5. 실행
x_input = np.array([5, 6, 7])
# print(x_input.shape)
x_input = x_input.reshape(1, 3, 1)
print(x_input)

y_hat = model.predict(x_input)
print(y_hat)
'''