import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# 1. 데이터
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0, 1, 1, 0])

print(x_data.shape)
print(y_data.shape)


# 2. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors = 1)
model = Sequential()

model.add(Dense(1000, input_shape = (2, )))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


# 3. 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_data, y_data, epochs = 100, batch_size = 1)


# 4. 평가 예측
res = model.evaluate(x_data, y_data)

x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_predict = model.predict(x_test)

# acc = accuracy_score([0, 1, 1, 0], y_predict)       # keras의 evaluate와 동일하다고 보면 됨; accuracy_score

# print(x_test, "의 예측 결과 : ", y_predict)
# print("acc : ", acc)

print("acc : ", res[1])
print(y_predict)