# 1. 사전 준비
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# 1-1. 필요한 함수 구현
# 1) RMSE 구현하기
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 2. 데이터 구성
x = np.array([range(1, 101), range(311, 411), range(100)])
y = np.array(range(711, 811))

# 2-1. 구성된 데이터 확인하기
# print("x : ", x)
# print("y : ", y)
# print(x.shape)
# -> 확인 결과, (3, 100)의 2차원 데이터로 확인


# 2-2. 데이터의 행과 열 전치시키기
x = x.transpose()
y = y.transpose()
# -> (100, 3)의 2차원 데이터로 전치시킴
# print(x)
# print(y)
# print(x.shape)
# print(y.shape)


# 2-3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2,
                                                    shuffle = True,
                                                    random_state = 1234)

# print(x_train)

# 3. DNN 신경망 모델 구성
model = Sequential()
model.add(Dense(200, input_dim = 3))
model.add(Dense(124))
model.add(Dense(1240))
model.add(Dense(1165))
model.add(Dense(1115))
model.add(Dense(2305))
model.add(Dense(1239))
model.add(Dense(913))
model.add(Dense(831))
model.add(Dense(1020))
model.add(Dense(983))
model.add(Dense(801))
model.add(Dense(918))
model.add(Dense(2320))
model.add(Dense(2011))
model.add(Dense(81))
model.add(Dense(9))
model.add(Dense(1))

# 4. 컴파일링 및 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse'])
model.fit(x_train, y_train,
          batch_size = 5,
          epochs = 50,
          validation_split = 0.25)

# 5. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test,
                           batch_size = 1)

y_predict = model.predict(x_test)

# 5-1. 결괏값 출력
# 1) loss 및 mse 값 출력
print("loss : ", loss)
print("-" * 40)
print("mse : ", mse)
print("-" * 40)

# 2) 예측값 출력
print("y의 예측값 : \n", y_predict)
print("-" * 40)

# 3) RMSE, R2 값 출력
print("RMSE : ", RMSE(y_test, y_predict))
print("-" * 40)
print("R2 : ", r2_score(y_test, y_predict))


"""
** RESULT **
--------------------
    Result 1
- 조건
    1) train_size : 75%
    2) shuffle = False
    3) batch_size = 5
    4) epochs = 50
    5) validation data : 14% (train_size * 0.19)
    6) first hidden layer = 100
    -> 결과 : loss :  0.09911314249038697
              mse  :  0.09911315143108368
              RMSE :  0.31482189539075667
              R2   :  0.9980939841188956


    Result 2
- 조건
    1) train_size : 80%
    2) shuffle = True
    3) batch_size = 1
    4) epochs = 50
    5) validation data : 20%(train_size * 0.25)
    6) first hidden layer = 200
    -> 결과 : loss : 0.4042428195476532
              mse  : 0.4042428135871887
              RMSE : 0.6358282842825868
              R2   : 0.9993992814052307

    
    Result 3
- 조건
    1) train_size : 80%
    2) shuffle = True
    3) batch_size = 1
    4) epochs = 50
    5) validation data : 20%(train_size * 0.25)
    6) first hidden layer = 200
    7) label is modified.(n : 1)
"""