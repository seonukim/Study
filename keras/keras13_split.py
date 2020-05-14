# 1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = False)


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(500, input_dim = 1))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))


# 3. 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 1,
          validation_split = 0.25
          #validation_data = (x_val, y_val)
)


# 4. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
print(y_predict)


# 5. RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


# 6. R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

print(x_train)