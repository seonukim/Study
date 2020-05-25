# 2020.05.25 수업 내용 리뷰
'''
주요 수업 내용
1. 데이터가 하나 주어졌을 때, predict, x, y로 나누기
2. 이를 이용하여 LSTM, Dense 모델링
3. matplotlib을 통해 주요 지표 시각화
'''

# 1. 사전 준비
from keras.models import Sequential, Model
from keras.layers import LSTM, Input, Dense
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 1-1. 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
tb_hist = TensorBoard(log_dir = './graph', histogram_freq = 0,
                      write_graph = True, write_images = True)

# 1-2. split 함수 생성
def split_data(a, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)


# 2. 데이터 준비
a = np.array(range(1, 101))
# print(a)

# 2-1. 데이터 분할
# 1) 예측용 데이터는 마지막 6행
# 2) train, test의 8:2로 분리
# 3) validation은 train의 20%
df = split_data(a, 5)
# print(df)
# print(df.shape)

# 2-2. predict 데이터 분할 (하단 6행)
x_pred = df[90: , :4]
print("=" * 40)
# print(x_pred)
# print(x_pred.shape)

# 2-3. x, y로 분할하기
x = df[ :90, :4]
# print("=" * 40)
# print(x.shape)

y = df[ :90, -1: ]
# print("=" * 40)
# print(y.shape)

x = x.reshape(90, 4, 1)
# print("=" * 40)
# print(x.shape)

x_pred = x_pred.reshape(6, 4, 1)
# print(x_pred)


# 2-4. train / test로 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True,
    random_state = 1234)
print("=" * 40)
print(x_train.shape)
print("=" * 40)
print(y_train.shape)


# 3. 모델링(LSTM)
model = Sequential()
model.add(LSTM(10, input_shape = (4, 1)))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(16))
model.add(Dense(18))
model.add(Dense(18))
model.add(Dense(18))
model.add(Dense(18))
model.add(Dense(18))
model.add(Dense(18))
model.add(Dense(1))

model.summary()


# 4. 실행 및 훈련
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
hist = model.fit(x_train, y_train,
                 batch_size = 1, epochs = 1000,
                 validation_split = 0.25, verbose = 2,
                 callbacks = [es])
print(hist)                     # 시각화 요소; <keras.callbacks.callbacks.History object at 0x0000015D0DB39208>
print(hist.history.keys())      # dict_keys(['val_loss', 'val_mse', 'loss', 'mse'])
print(hist.history.items())     # dictionary 형식으로 보여줌


# 5. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test)
print("=" * 40)
print("loss : ", loss)
print("mse : ", mse)

pred = model.predict(x_pred, batch_size = 1)
print("=" * 40)
print("pred : \n", pred) 

# 6. 주요 지표 시각화
plt.plot(hist.history['loss'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()