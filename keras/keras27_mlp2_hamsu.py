# keras14_mlp Sequential 모델을 함수형으로 변경
# early_stopping 적용

# 1. 사전 준비
# 1-1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

# 1-2. 필요 함수 정의
# 1-2-1. RMSE 함수 정의
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 2. 데이터 준비
x = np.array([range(1, 101), range(311, 411), range(100)])
y = np.array(range(711, 811))

# print(x.shape)
# print(y.shape)

# 2-1. 데이터 전치
x = np.transpose(x)
y = np.transpose(y)

# print(x.shape)


# 2-2. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True,
    random_state = 1234)

# print(x_train.shape)


# 3. 모델 구성
# 3-1. Input 데이터
input1 = Input(shape = (3, ))
dense1 = Dense(50, activation = 'relu')(input1)
dense2 = Dense(46, activation = 'relu')(dense1)
dense3 = Dense(55, activation = 'relu')(dense2)
dense4 = Dense(40)(dense3)

# 3-2. Output 데이터
output1 = Dense(66)(dense4)
output2 = Dense(23)(output1)
output3 = Dense(45)(output2)
output4 = Dense(1)(output3)

# 3-3. 모델링
model = Model(inputs = input1, outputs = output4)

# 3-4. 모델 요약 확인
model.summary()


# 4. 훈련
# 4-1. 조기종료 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

# 4-2. 모델 컴파일
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

# 4-3. 모델 훈련
model.fit(x_train, y_train,
          epochs = 10000, batch_size = 10,
          validation_split = 0.25, verbose = 1,
          callbacks = [es])


# 5. 평가 및 예측
# 5-1. 모델 평가
result = model.evaluate(x_test, y_test, batch_size = 10)
print("=" * 40)
print("Result : ", result)

# 5-2. 결과 예측
y_predict = model.predict(x_test)
print("=" * 40)
print("y_predict : \n", y_predict)

# 6. 성능 지표 확인
# 6-1. RMSE
print("=" * 40)
print("RMSE : ", RMSE(y_test, y_predict))

# 6-2. R2
print("=" * 40)
print("R2 : ", r2_score(y_test, y_predict))
