# 1. 데이터 생성
# numpy 모듈을 import하여 변수 x, y에 numpy 배열을 할당
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 2. 모델 구성
# keras 모듈에서 Sequential(순차적 모델)과
# Dense(레이어, 노드 구성) 클래스 import
from keras.models import Sequential
from keras.layers import Dense

# 2_1. 레이어 및 노드 구성
# input_dim = 1 : 최초 레이어의 노드는 1
# 최초 레이어의 노드 1개를 받아서 첫번째 hidden layer에서 1을 출력
# 활성화 방법은 'relu' ; rectified linear unit
model = Sequential()
model.add(Dense(1, input_dim = 1, activation = 'relu'))

# 3. 컴파일 및 실행
# 코드를 머신이 이해할 수 있게끔 바꿔주는 과정을 compile이라고 함
# loss(손실 함수)는 mse ; mean_squared_error ; 평균 제곱 오차
# optimizer(최적화 함수)는 'adam' ; adaptive moment estimation
# metrics(판별 방식)는 'accuracy' ; 정확도
model.compile(loss = 'mean_squared_error', optimizer = 'adam',
              metrics = ['accuracy'])

# 4. 예측 실행
# model 적합, epochs(반복 횟수)는 500번, batch_size(몇개로 자를지)는 1
# model 평가, loss와 acc로 나타냄
model.fit(x, y, epochs = 500, batch_size = 1)
loss, acc = model.evaluate(x, y, batch_size = 1)

# 5. 결과 출력
print("loss : ", loss)
print("acc : ", acc)