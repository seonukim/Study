"""
!하이퍼 파라미터 튜닝
1. 노드의 갯수 조절
2. 레이어의 깊이 조절
3. epochs(반복 횟수) 조절
4. batch_size 조절(default = 32)
"""

# 1. 필요한 모듈 import
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 2. 데이터 준비
x = np.array([2, 4, 6, 8, 10])
y = np.array([4, 8, 12, 16, 20])

# 3. 딥러닝 모델링
model = Sequential()
model.add(Dense(9, input_dim = 1, activation = 'relu'))
model.add(Dense(31))
model.add(Dense(82))
model.add(Dense(124))
model.add(Dense(148))
model.add(Dense(181))
model.add(Dense(174))
model.add(Dense(157))
model.add(Dense(134))
model.add(Dense(94))
model.add(Dense(52))
model.add(Dense(23))
model.add(Dense(1, activation = 'relu'))

# 4. 컴파일링
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

# 5. 평가 및 예측
model.fit(x, y, epochs = 1000, batch_size = 1, validation_data = (x, y))

loss, acc = model.evaluate(x, y, batch_size = 1)

# 6. 결과 출력
print("loss : ", loss)
print("acc : ", acc)

# 결과(1)
# loss : 1.9213075574953107e-12
# acc  : 1.0