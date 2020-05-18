""" ensemble 모델 구현 """

# 1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(311, 411), range(411, 511)])
x2 = np.array([range(711, 811), range(711, 811), range(511, 611)])

y1 = np.array([range(101, 201), range(411, 511), range(100)])

##################################
##### 여기서부터 수정하세요. #####
##################################

# 1-1. 행과 열을 바꾸기 - 전치행렬 구하기
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

print(x1)
print(x2)
print(y1)



# 1-2. 데이터 분할
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, test_size = 0.2, shuffle = False)

y_train, y_test = train_test_split(
    y1, test_size = 0.2, shuffle = False)

# print(x1_train.shape)
# print("-" * 40)
# print(x2_train)
# print("-" * 40)
# print(x1_test)
# print("-" * 40)
# print(x2_test)
# print("-" * 40)
# print(y1_train)
# print("-" * 40)
# print(y1_test)



# 2. 모델 구성 - 함수형 모델
from keras.models import Sequential, Model     # Model = 함수형 모델
from keras.layers import Dense, Input          # Input = 함수형 모델 인풋 레이어

# 2-1. 함수형 모델의 첫번째 모델
"""  summary() 시 출력되는 레이어의 이름 변경 ; name = '변경할 이름' """
input1 = Input(shape = (3, ))    # 첫번째 인풋 레이어 구성 후 input1 변수에 할당
dense1_1 = Dense(708, activation = 'relu')(input1)  # 첫번째 모델의 첫번째 히든레이어 구성
dense1_2 = Dense(570, activation = 'relu')(dense1_1)  # 첫번째 모델의 두번째 히든레이어 구성
dense1_3 = Dense(401, activation = 'relu')(dense1_2)

# 2-2. 함수형 모델의 두번째 모델
input2 = Input(shape = (3, ))
dense2_1 = Dense(702, activation = 'relu')(input2)
dense2_2 = Dense(430, activation = 'relu')(dense2_1)
dense2_3 = Dense(404, activation = 'relu')(dense2_2)

# 2-3. 모델 병합
from keras.layers.merge import concatenate       # 모델 병합 모듈 임포트 - concatenate ; '잇다, 일치시키다'
merge1 = concatenate([dense1_3, dense2_3])    # 각 모델의 마지막 레이어 입력
middle1 = Dense(140)(merge1)
middle2 = Dense(230)(middle1)
middle3 = Dense(301)(middle2)
middle4 = Dense(310)(middle3)

# 2-4. 각 모델의 output레이어 구성
output1 = Dense(180)(middle4)
output2 = Dense(310)(output1)
output3 = Dense(3)(output2)

model = Model(inputs = [input1, input2],
              outputs = output3)   # 함수형 병합 모델 구성(인풋, 아웃풋 명시)

model.summary()  # 모델 요약표



# 3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1_train, x2_train], y_train,
          epochs = 80, batch_size = 1, validation_split = 0.25, verbose = 3)


# 4. 평가 및 예측
loss, mse = model.evaluate([x1_test, x2_test], y_test)
print("-" * 40)
print("-" * 40)
print("loss : ", loss)   # 병합된 전체 모델의 loss
print("-" * 40)
print("mse : ", mse)   # 아웃풋1에 대한 loss
print("-" * 40)

y_predict = model.predict([x1_test, x2_test])
print(y_predict)
print("-" * 40)




# 5. RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print("-" * 40)
# rmse2 = RMSE(y2_test, y2_predict)
# # print("RMSE_2 : ", rmse2)
# # print("-" * 40)
# rmse3 = RMSE(y3_test, y3_predict)
# # print("RMSE_3 : ", rmse3)
# # print("-" * 40)


# 6. R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
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
print("R2 : ", r2)
