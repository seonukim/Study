# 1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_pred = np.array([11, 12, 13])
# predict

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense   # DNN(Deep Neural Network) 구조의 가장 기본적인 구조
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

# 3. 훈련
# metrics의 인자값을 ['acc'] -> ['mse']로 수정
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y, epochs = 30, batch_size = 1)

# 4. 평가 및 예측
# 동일하게, acc를 mse로 수정
# evaluate() -> 이전에 fit()에서 이미 훈련한 데이터로 평가를 진행하고 있다.
# 이럴 경우, 훈련용 데이터와 평가용 데이터를 구분하여 평가를 진행해야 한다.
loss, mse = model.evaluate(x, y, batch_size = 1)
print("loss : ", loss)
print("mse : ", mse)

y_pred = model.predict(x_pred)
print("y_predict : ", y_pred)
