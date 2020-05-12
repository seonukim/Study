#1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense  # 1차함수라고 생각

#2_1. 레이어 및 노드 구성
model = Sequential()
model.add(Dense(500, input_dim = 1))
model.add(Dense(430))
model.add(Dense(920))
model.add(Dense(453))
model.add(Dense(182))
model.add(Dense(349))
model.add(Dense(890))
model.add(Dense(997))
model.add(Dense(1))

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x, y, epochs = 100)

#4. 평가예측
loss, acc = model.evaluate(x, y)
print("acc : ", acc)

y_predict = model.predict(x2)
print(y_predict)


# 하이퍼파라미터 튜닝
# 1. epochs(반복 횟수) 조절
# 2. 레이어의 깊이 조절
# 3. 노드의 갯수 조절
# 4. batch_size 조절(default = 32)