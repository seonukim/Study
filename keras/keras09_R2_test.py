"""
2020.05.13 과제
R2를 음수가 아닌 0.5 이하로 줄이기.
레이어는 인풋과 아웃풋을 포함하여 5개 이상,
노드는 레이어 당 각각 5개 이상
batch_size = 1
epochs = 100 이상
"""

# 1. 데이터
import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15])
y_test = np.array([11, 12, 13, 14, 15])
x_pred = np.array([16, 17, 18])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1, activation = 'sigmoid'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x_train, y_train, epochs = 500, batch_size = 1, validation_data = (x_test, y_test))

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


"""
훈련 데이터와 같은 데이터로 평가를 했을 때 : 0.82
compile의 parameter 중 loss를 'binary_crossentropy'로 했을 때 : 0.07 -> binary는 1, 0의 이진분류법인데
회귀 모델에 적용하니 설명력이 낮게 나오는 것 같다.
activation의 인자값을 sigmoid 계열로 입력 시 -> sigmoid계열 역시 분류함수인데 회귀 모델에 적용..
"""

"""
레이어의 수를 늘리고 노드를 일괄적으로 10으로 세팅
인풋 레이어의 활성화 함수 'sigmoid' - 분류 유형
아웃풋 레이어의 활성화 함수 'relu'  - 선형 유형
손실함수 'mse'

>>> R2 :  0.09246565130943052
"""

"""
레이어 및 노드 조절
인풋 레이어 활성화 함수 'sigmoid' - 분류 유형
손실함수 'mse'

>>> R2 :  0.1157790464274513
"""

"""
바로 위와 같은 조건
epochs만 500회로 조정

>>> R2 :  0.2753925863813492
"""

"""
바로 위와 같은 조건
epochs를 1000회로 조정

>>> R2 :  0.4917325451944634
"""

"""
레이어 및 노드를 5개씩으로 세팅
epochs를 500회로 조정

>>> R2 : 0.20131958691945329
"""