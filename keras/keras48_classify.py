'''20200526 분류모델'''

# 회귀모델을 먼저 구현해보자.
# 1. 사전준비
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
import numpy as np

# 1-1. 조기종료 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

# 2. 데이터
x = np.array(range(1, 11))
y = np.array([1, 0 ,1, 0, 1, 0, 1, 0, 1, 0])
print(x.shape)


# 3. 모델 구성
model = Sequential()
model.add(Dense(10, input_shape = (1, ), activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

# input1 = Input(shape = ())

model.summary()

# 4. 실행 및 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y, epochs = 1000,
                 batch_size = 1, callbacks = [es])

# 5. 평가 및 예측
loss, mse = model.evaluate(x, y)
print("loss : ", loss)
print("mse : ", mse)

pred = model.predict(x)
print("pred : ", pred)