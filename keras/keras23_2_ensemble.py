""" 2020.05.18 수업 내용 복습하기 """
""" 다 : 다 모델 """

# 1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

# 2. 데이터 구성
x1 = np.array([range(1, 101), range(311, 411)])
x2 = np.array([range(711, 811), range(711, 811)])

y1 = np.array([range(101, 201), range(411, 511)])
y2 = np.array([range(501, 601), range(711, 811)])
y3 = np.array([range(411, 511), range(611, 711)])

# 2-1. 데이터 전치행렬 구하기 ((2, 100) -> (100, 2))
x1 = x1.transpose()
x2 = x2.transpose()

y1 = y1.transpose()
y2 = y2.transpose()
y3 = y3.transpose()

# print(x1)

# 2-2. 데이터 분할하기
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, test_size = 0.2, shuffle = False)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, test_size = 0.2, shuffle = False)

y3_train, y3_test = train_test_split(
    y3, test_size = 0.2, shuffle = False)

# 3. 모델링
input1 = Input(shape = (2, ))
dense1 = Dense(500, activation = 'relu')(input1)
dense1_2 = Dense(400, activation = 'relu')(dense1)
dense1_3 = Dense(250)(dense1_2)

input2 = Input(shape = (2, ))
dense2 = Dense(400, activation = 'relu')(input2)
dense2_2 = Dense(300, activation = 'relu')(dense2)
dense2_3 = Dense(150, activation = 'relu')(dense2_2)

merge = concatenate([dense1_3, dense2_3])
middle = Dense(100)(merge)
middle = Dense(80)(middle)
middle = Dense(50)(middle)

output1 = Dense(200)(middle)
output1_2 = Dense(150)(output1)
output1_3 = Dense(2)(output1_2)

output2 = Dense(200)(middle)
output2_2 = Dense(150)(output2)
output2_3 = Dense(2)(output2_2)

output3 = Dense(200)(middle)
output3_2 = Dense(150)(output3)
output3_3 = Dense(2)(output3_2)

model = Model(inputs = [input1, input2],
              outputs = [output1_3, output2_3, output3_3])

# model.summary()


# 4. 컴파일 및 훈련
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1_train, x2_train],
          [y1_train, y2_train, y3_train],
          epochs = 5, batch_size = 1,
          validation_split = 0.25, verbose = 1,
          callbacks = [es])

# 5. 평가 및 예측
res = model.evaluate([x1_test, x2_test],
                     [y1_test, y2_test, y3_test],
                     batch_size = 1)
# print("Result : ", res)

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
print("=" * 40)
print("y1_pred : \n", y1_predict)
print("=" * 40)
print("y2_pred : \n", y2_predict)
print("=" * 40)
print("y3_pred : \n", y3_predict)



# 6. RMSE
def RMSE(y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))

def mean_result(list):
    return (sum(list) / len(list))

rmse1 = RMSE(y1_test, y1_predict)
rmse2 = RMSE(y2_test, y2_predict)
rmse3 = RMSE(y3_test, y3_predict)
a = [rmse1, rmse2, rmse3]
print("=" * 40)
print("RMSE : ", mean_result(a))
print("=" * 40)
# print("RMSE : ", (rmse1 + rmse2 + rmse3) / 3)

# 7. R2
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2_3 = r2_score(y3_test, y3_predict)
b = [r2_1, r2_2, r2_3]

print("R2 : ", mean_result(b))