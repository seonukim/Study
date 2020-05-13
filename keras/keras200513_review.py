# 1. 모듈 임포트
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# 1_1. RMSE 함수 생성
# RMSE? -> Root Mean Squared Error의 약자로, MSE에 Root를 씌운 것이다.
# np.sqrt() method를 이용하여 RMSE()라는 함수를 정의하여 사용한다.
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 2. 데이터 구성
# 데이터는 훈련용 셋, 평가용 셋, 예측용 셋으로 구분한다.
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15])
y_test = np.array([11, 12, 13, 14, 15])
x_pred = np.array([16, 17, 18])

# 3. 모델링
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# 3_1. 각 레이어의 파라미터 값 확인
model.summary()

# 4. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

# 5. 평가 및 예측
# 5_1. 예측값 출력
y_predict = model.predict(x_test)
print("y_predict : ", y_predict)

# 5_2. loss, mse 출력
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse : ", mse)

# 5_3. RMSE 출력
print("RMSE : ", RMSE(y_test, y_predict))

# 5_4. R2 출력
# R2? -> r_squared라고 하며, 설명력, 결정계수 등으로 불린다.
print("R2 : ", r2_score(y_test, y_predict))