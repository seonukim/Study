# ensemble 모델 구현

# 1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(311, 411), range(100)])
y1 = np.array([range(711, 811), range(711, 811), range(100)])

x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

# 1-1. 행과 열을 바꾸기 - 전치행렬 구하기
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

# 1-2. 데이터 분할
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, test_size = 0.2, shuffle = False)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, test_size = 0.2, shuffle = False)


# 2. 모델 구성 - 함수형 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# model.add(Dense(5, input_dim = 3))
# model.add(Dense(4))
# model.add(Dense(1))

# 2-1. 함수형 모델의 첫번째 모델
""" summary() 시 출력되는 레이어의 이름 변경 ; name = '변경할 이름'"""
input1 = Input(shape = (3, ))    # 첫번째 인풋 레이어 구성 후 input1 변수에 할당
dense1_1 = Dense(30, activation = 'relu')(input1)  # 첫번째 모델의 첫번째 히든레이어 구성
dense1_2 = Dense(50, activation = 'relu')(dense1_1)  # 첫번째 모델의 두번째 히든레이어 구성
dense1_3 = Dense(43, activation = 'relu')(dense1_2)
dense1_4 = Dense(12, activation = 'relu')(dense1_3)
dense1_5 = Dense(4)(dense1_4)

# 2-2. 함수형 모델의 두번째 모델
input2 = Input(shape = (3, ))
dense2_1 = Dense(51, activation = 'relu')(input2)
dense2_2 = Dense(43, activation = 'relu')(dense2_1)
dense2_3 = Dense(44, activation = 'relu')(dense2_2)
dense2_4 = Dense(23, activation = 'relu')(dense2_3)
dense2_5 = Dense(2)(dense2_4)

# 2-3. 모델 병합
from keras.layers.merge import concatenate       # 모델 병합 모듈 임포트 - concatenate ; '잇다, 일치시키다'
merge1 = concatenate([dense1_5, dense2_5])    # 각 모델의 마지막 레이어 입력
middle1 = Dense(10)(merge1)
middle2 = Dense(21)(middle1)
middle3 = Dense(34)(middle2)
middle4 = Dense(5)(middle3)

# 2-4. 각 모델의 output레이어 구성
output1 = Dense(10)(middle4)
output1_2 = Dense(30)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(10)(middle4)
output2_2 = Dense(30)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs = [input1, input2],
              outputs = [output1_3, output2_3])   # 함수형 병합 모델 구성(인풋, 아웃풋 명시)

model.summary()  # 모델 요약표


# 3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1_train, x2_train],
          [y1_train, y2_train],
          epochs = 50, batch_size = 1)
        #   validation_split = 0.25, verbose = 1)


# 4. 평가 및 예측
loss1, loss2, loss3, mse1, mse2 = model.evaluate([x1_test, x2_test],
                                                 [y1_test, y2_test], batch_size = 1)
print("loss1 : ", loss1)   # 병합된 전체 모델의 loss
print("loss2 : ", loss2)   # 모델1에 대한 loss
print("loss3 : ", loss3)   # 모델2에 대한 loss
print("mse1 : ", mse1)     # 모델1에 대한 mse
print("mse2 : ", mse2)     # 모델2에 대한 mse

# y_predict = model.predict([x1_test, x2_test])
# print(y_predict)


# # 5. RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error([y1_test, y2_test], y_predict))
# print("RMSE : ", RMSE([y1_test, y2_test], y_predict))


# # 6. R2 구하기
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)
