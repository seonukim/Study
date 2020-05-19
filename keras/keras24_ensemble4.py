""" ensemble 모델 구현 """

# 1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(301, 401)])

y1 = np.array([range(711, 811), range(611, 711)])
y2 = np.array([range(101, 201), range(411, 511)])

##################################
##### 여기서부터 수정하세요. #####
##################################

# 1-1. 행과 열을 바꾸기 - 전치행렬 구하기
x1 = np.transpose(x1)

y1 = np.transpose(y1)
y2 = np.transpose(y2)

# print(x1)

# print(y1)
# print(y2)


# 1-2. 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(
    x1, test_size = 0.2, shuffle = False)

y1_train, y1_test, y2_train, y2_test = train_test_split(
    y1, y2, test_size = 0.2, shuffle = False)

# print(x_train)
# print(x_test)


# 2. 모델 구성 - 함수형 모델
from keras.models import Sequential, Model     # Model = 함수형 모델
from keras.layers import Dense, Input          # Input = 함수형 모델 인풋 레이어

# 2-1. 함수형 모델의 첫번째 모델
"""  summary() 시 출력되는 레이어의 이름 변경 ; name = '변경할 이름' """
input1 = Input(shape = (2, ))    # 첫번째 인풋 레이어 구성 후 input1 변수에 할당
dense1_1 = Dense(18, activation = 'relu')(input1)  # 첫번째 모델의 첫번째 히든레이어 구성
dense1_2 = Dense(17, activation = 'relu')(dense1_1)  # 첫번째 모델의 두번째 히든레이어 구성
dense1_3 = Dense(18, activation = 'relu')(dense1_2)
dense1_4 = Dense(21)(dense1_3)


# 2-2. 모델 병합
# from keras.layers.merge import concatenate       # 모델 병합 모듈 임포트 - concatenate ; '잇다, 일치시키다'
# merge1 = concatenate(dense1_4)    # 각 모델의 마지막 레이어 입력
# middle1 = Dense(29)(merge1)
# middle2 = Dense(33)(middle1)
# middle3 = Dense(14)(middle2)

# 2-3. 각 모델의 output레이어 구성
output1 = Dense(31)(dense1_4)
output1_2 = Dense(37)(output1)
output1_3 = Dense(2)(output1_2)

output2 = Dense(31)(dense1_4)
output2_2 = Dense(37)(output1)
output2_3 = Dense(2)(output2_2)

model = Model(inputs = input1,
              outputs = [output1_3, output2_3])   # 함수형 병합 모델 구성(인풋, 아웃풋 명시)

model.summary()  # 모델 요약표



# 3. 훈련
# 과적합 방지 - 조기종료(EarlyStopping) 기법 사용
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train,
          [y1_train, y2_train],
          epochs = 200, batch_size = 1,
          validation_split = 0.25, verbose = 1,
          callbacks = [es])


# 4. 평가 및 예측
result = model.evaluate(x_test, [y1_test, y2_test])
print("=" * 40)
print("Result : ", result)   # 병합된 전체 모델의 loss
print("=" * 40)

y1_predict, y2_predict = model.predict(x_test)
print(y1_predict)
print(y2_predict)
print("=" * 40)




# 5. RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse1 = RMSE(y1_test, y1_predict)
rmse2 = RMSE(y2_test, y2_predict)
print("RMSE : ", (rmse1 + rmse2) / 2)
print("=" * 40)
# rmse2 = RMSE(y2_test, y2_predict)
# # print("RMSE_2 : ", rmse2)
# # print("-" * 40)
# rmse3 = RMSE(y3_test, y3_predict)
# # print("RMSE_3 : ", rmse3)
# # print("-" * 40)


# 6. R2 구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
# r2_2 = r2_score(y2_test, y2_predict)
# r2_3 = r2_score(y3_test, y3_predict)
# print("R2_1 : ", r2_1)
# print("-" * 40)
# print("R2_2 : ", r2_2)
# print("-" * 40)
# print("R2_3 : ", r2_3)
# print("-" * 40)
# print("RMSE : ", (rmse1 + rmse2 + rmse3) / 3)
# print("-" * 40)
print("R2 : ", (r2_1 + r2_2) / 2)
