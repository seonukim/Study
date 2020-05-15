'''
    - 2020.05.15 실습 -
1. R2 0.5 이하
2. layers는 5개 이상
3. nodes 10개 이상
4. batch_size = 8 이하
5. epochs = 30 이상
'''


# 1. 사전 준비
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# 1-1. 필요 함수 구현
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


# 2. 데이터 구성
x = np.array([range(1, 101), range(311, 411), range(100)])
y = np.array(range(711, 811))

# 2-2. 데이터 전치시키기
x = x.transpose()
y = y.transpose()

# 2-3. 데이터 분할하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 1234)


# 3. 모델 구성
model = Sequential()
model.add(Dense(10000, input_dim = 3))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(10000))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(10000))
model.add(Dense(1))


# 4. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, batch_size = 25, epochs = 50, validation_split = 0.25, verbose = 2)


# 5. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 20)
y_predict = model.predict(x_test)

# 6. 결괏값 출력
print("-" * 40)
print("loss : ", loss)
print("-" * 40)
print("mse : ", mse)
print("-" * 40)
print("RMSE : ", RMSE(y_test, y_predict))
print("-" * 40)
print("R2 : ", r2_score(y_test, y_predict))
print("-" * 40)
print("y의 예측값 : \n", y_predict)