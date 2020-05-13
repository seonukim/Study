# 1. 데이터
# 훈련용 데이터와 평가용 데이터를 구분하기 위해
# x_train, y_train, x_test, y_test로 만들어준다.
import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15])
y_test = np.array([11, 12, 13, 14, 15])
x_pred = np.array([16, 17, 18])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense   # DNN(Deep Neural Network) 구조의 가장 기본적인 구조
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(21))
model.add(Dense(30))
model.add(Dense(42))
# model.add(Dense(132))
# model.add(Dense(184))
# model.add(Dense(235))
# model.add(Dense(330))
# model.add(Dense(248))
# model.add(Dense(349))
# model.add(Dense(200))
# model.add(Dense(180))
model.add(Dense(36))
model.add(Dense(24))
model.add(Dense(1))

# 3. 훈련
# metrics의 인자값을 ['acc'] -> ['mse']로 수정
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 80, batch_size = 2)

# 4. 평가 및 예측
# 동일하게, acc를 mse로 수정
# evaluate() -> 이전에 fit()에서 이미 훈련한 데이터로 평가를 진행하고 있다.
# 이럴 경우, 훈련용 데이터와 평가용 데이터를 구분하여 평가를 진행해야 한다.
loss, mse = model.evaluate(x_test, y_test, batch_size = 2)
print("loss : ", loss)
print("mse : ", mse)

y_pred = model.predict(x_pred)
print("y_predict : ", y_pred)

"""
# 결과(1) - 레이어 줄이기 및 노드 조절 - 베스트는 아니지만 예측값이 가장 근사함
# loss :  3.019522409886122e-10
# mse  :  3.019522409886122e-10
# y_predict :  [[16.000015]
#               [17.000023]
#               [18.000048]]
"""
# 결과(2) - 레이어 더 줄이기
# loss :  1.0456642485223711e-08
# mse  :  1.0456642662859394e-08
# y_predict :  [[15.999775]
#               [16.999723]
#               [17.99968 ]]

# 결과(3) - 레이어 더 줄이기
# loss :  0.00034816157713066784
# mse  :  0.000348161585861817
# y_predict :  [[16.023672]
#               [17.025398]
#               [18.027126]]
"""
# 결과(4) - 레이어 줄이기 및 에포 조절 - 베스트
# loss :  3.092281986027956e-12
# mse  :  3.09228207276413e-12
# y_predict :  [[16.000002]
#               [16.999998]
#               [17.999996]]
"""
# 결과(5) - 에포 조절
# loss :  0.5954108357429504
# mse  :  0.5954108238220215
# y_predict :  [[16.948416]
#               [18.008936]
#               [19.06945 ]]

# 결과(6)
# loss :  5.086507189844269e-06
#  mse :  5.086506916995859e-06
# y_predict :  [[15.996422]
#               [16.99595 ]
#               [17.99547 ]]