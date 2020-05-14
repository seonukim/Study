# 1. 모듈 임포트
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# 2. 데이터 구성
x = np.array(range(2, 501, 2))
y = np.array(range(2, 501, 2))

# print(x)
# print(y)

# 2-1. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.25,
                                                    random_state = 1234,
                                                    shuffle = True)

# print(x_test)
# print(y_test)

# 3. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim = 1, activation = 'relu'))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(250))
model.add(Dense(300))
model.add(Dense(350))
model.add(Dense(400))
model.add(Dense(350))
model.add(Dense(300))
model.add(Dense(250))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(1))

# 3-1. 모델 요약 확인
model.summary()

# 4. 컴파일 및 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(x_train, y_train,
          batch_size = 1,
          epochs = 100,
          validation_split = 0.2)

# 5. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
print(y_predict)

# 6. RMSE
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

# 7. R2
print("R2 : ", r2_score(y_test, y_predict))