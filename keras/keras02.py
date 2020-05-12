# 1. 패키지 로드
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 2. 데이터 구성
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
y_test = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

# 3. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1, activation = 'relu'))

model.summary()

# 4. 컴파일하기
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 5. 모델 적합 및 예측
model.fit(x_train, y_train,
          epochs = 500,
          batch_size = 1,
          validation_data = (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)

# 6. 결과 출력
print("loss : ", loss)
print("acc : ", acc)

# 7. 결과물 확인
output = model.predict(x_test)
print("결과물 : \n", output)