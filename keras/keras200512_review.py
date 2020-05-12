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
# node 갯수 조절
# layer의 depth 조절 - 줄이기
# layer의 depth 조절 - 늘리기
model = Sequential()
model.add(Dense(91, input_dim = 1, activation = 'relu'))
model.add(Dense(111))
model.add(Dense(131))
model.add(Dense(181))
model.add(Dense(201))
model.add(Dense(264))
model.add(Dense(246))
model.add(Dense(187))
model.add(Dense(154))
model.add(Dense(151))
model.add(Dense(148))
model.add(Dense(134))
model.add(Dense(110))
model.add(Dense(86))
model.add(Dense(43))
model.add(Dense(1, activation = 'relu'))

# 4. 컴파일링
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

# 5. 평가 및 예측
# epochs 조절 ; 1000 -> 100
model.fit(x, y, epochs = 100, batch_size = 1, validation_data = (x, y))

loss, acc = model.evaluate(x, y, batch_size = 1)

# 6. 결과 출력
print("loss : ", loss)
print("acc : ", acc)

# 결과(1)
# loss : 1.9213075574953107e-12
# acc  : 1.0

# 결과(2)
# loss : 1.2038549220960703e-05
# acc  : 1.0

# 결과(3)
# loss : 2.276204486406641e-06
# acc  : 1.0

# 결과(4)
# loss : 0.353423085808754
# acc  : 0.4000000059604645