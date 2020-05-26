# 1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 1-1. 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
enc = OneHotEncoder()

# 2. 데이터
x = np.array(range(1, 11))
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]).reshape(-1, 1)
enc.fit(y)
y = enc.transform(y).toarray()
print(x.shape)
print(y.shape)

# 3. 모델 구성
model = Sequential()
model.add(Dense(100, input_shape = (1, ), activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))

model.summary()

# 4. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['accuracy'])
model.fit(x, y, epochs = 100, batch_size = 1)

# 5. 평가 및 예측
loss, acc = model.evaluate(x, y, batch_size = 1)
print("loss : ", loss)
print("acc : ", acc)

x_pred = np.array([1, 2, 3, 4, 5])
y_pred = model.predict(x_pred)
print("y_pred : \n", y_pred)
