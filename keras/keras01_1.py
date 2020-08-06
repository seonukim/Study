#1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  # 1차함수라고 생각

#2_1. 레이어 및 노드 구성
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x, y, epochs = 100, batch_size = 1)

#4. 평가예측
loss, acc = model.evaluate(x, y, batch_size = 1)
print("acc : ", acc)