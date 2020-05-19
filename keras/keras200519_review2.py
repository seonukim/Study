'''
MLP - Multi Layer Perceptron; 다층 퍼셉트론
함수형 모델로 구현
두번째 : 각 데이터의 입력 차원이 다른 경우, input = 3, output = 1
print()의 모든 결과는 해당 코드 오른쪽에 res로 표기함.
'''

# 1. 사전 준비
# 1-1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

# 1-2. 필요 함수 및 객체 생성
# 1-2-1. RMSE 함수 정의
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 1-2-2. EarlyStopping 클래스 객체 생성
early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)


# 2. 데이터 구성하기
x = np.array([range(1, 101), range(311, 411), range(100)])
y = np.array(range(711, 811))
print("=" * 40)
print("배열 x의 차원 : ", x.shape)      # (3, 100)
print("=" * 40)
print("배열 y의 차원 : ", y.shape)      # (100, )

# 2-1. 데이터 차원 전치시키기
x = x.transpose()
y = y.transpose()
print("=" * 40)
print("배열 x의 차원 : ", x.shape)      # (100, 3)
print("=" * 40)
print("배열 y의 차원 : ", y.shape)      # (100, )

# 2-2. 데이터 분할하기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True,
    random_state = 1234)
print("=" * 40)
print("=" * 40)
print("훈련용 데이터 x의 차원 : ", x_train.shape)       # (80, 3)
print("=" * 40)
print("훈련용 데이터 y의 차원 : ", y_train.shape)       # (80, )
print("=" * 40)
print("평가용 데이터 x의 차원 : ", x_test.shape)        # (20, 3)
print("=" * 40)
print("평가용 데이터 y의 차원 : ", y_test.shape)        # (20, )


# 3. 모델 구성
# 3-1. Input Layer 구성
input1 = Input(shape = (3, ))
dense1 = Dense(50, activation = 'relu')(input1)
dense2 = Dense(45, activation = 'relu')(dense1)
dense3 = Dense(40)(dense2)

# 3-2. Output Layer 구성
output1 = Dense(38)(dense3)
output2 = Dense(34)(output1)
output3 = Dense(1)(output2)     # 출력 데이터가 (100, ) 이기 때문에

# 3-3. 모델링
model = Model(inputs = input1,
              outputs = output3)

# 3-4. 모델 요약표
model.summary()


# 4. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train,
          epochs = 10000, batch_size = 10,
          validation_split = 0.25, verbose = 3,
          callbacks = [early])

# 5. 평가 및 예측
# 5-1. 모델 평가
result = model.evaluate(x_test, y_test)
print("=" * 40)
print("Result : ", result)      # res : [1.1407562494277954, 1.1407562494277954]

# 5-2. 결과 예측
y_predict = model.predict(x_test)
print("=" * 40)
print("예측값 : \n", y_predict)     # (100, ) 의 차원을 가진 y의 예측값 출력


# 6. 성능 지표 확인
# 6-1. RMSE 확인
print("=" * 40)
print("RMSE : ", RMSE(y_test, y_predict))       # res : 0.3756902059990046

# 6-2. R2 확인
print("=" * 40)
print("R2 : ", r2_score(y_test, y_predict))     # res : 0.9997902745495719