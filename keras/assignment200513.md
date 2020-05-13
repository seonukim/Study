## 2020.05.13 과제에 관한 나의 생각

---

####과제
R2를 0 < R2 <= 0.5로 줄이자.
레이어는 인풋과 아웃풋을 포함하여 5개 이상, 노드는 레이어 당 5개 이상.
batch_size = 1로 고정
epochs = 100 이상
```
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

model.add(Dense(5, input_dim = 1, activation='hard_sigmoid'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data = (x_test, y_test))

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
```

위와 같은 코드에서,<br/>
`Dense()` 메서드의 파라미터 중 활성화 함수 파라미터인 **activation**의 값을

선형 활성화 또는 회귀 계열 함수로 넣지 않고, 분류 계열 함수,

예를 들어 **sigmoid or hard_sigmoid**로 입력했을 때,<br/>
우리에게 주어진 데이터는 **y = wx + b** 모양의 함수를 갖는 ++**선형적 데이터**++ 인데
활성화 함수에 분류 계열의 함수를 넣어서 결정 계수의 값이 낮아지는게 아닐까 싶다.
<br/>
또 한 가지는 `model.compile()` 부분인데, `compile()` 메서드의 파라미터 중

**loss(손실함수)**의 인자 값을 선형 또는 회귀 계열이 아닌 분류 계열의 손실 함수를 입력했을 때
위의 생각과 같은 이유로 결정 계수의 값이 낮아지는게 아닐까 싶다.

분류 계열 손실함수의 예 : **binary_crossentropy**


[활성화 함수 사용의 예; Sigmoid 대신 ReLU? 상황에 맞는 활성화 함수 사용하기](https://medium.com/@kmkgabia/ml-sigmoid-%EB%8C%80%EC%8B%A0-relu-%EC%83%81%ED%99%A9%EC%97%90-%EB%A7%9E%EB%8A%94-%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-c65f620ad6fd)

[[손실함수]Binary Cross Entropy](https://curt-park.github.io/2018-09-19/loss-cross-entropy/)