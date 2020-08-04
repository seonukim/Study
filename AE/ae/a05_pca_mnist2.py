'''
keras56_mnist_DNN.py 땡겨라
input_dim = 154로 모델을 만드시오
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(x_test.shape) # (10000, 28, 28)
print(y_train.shape) # (60000,)
print(y_test.shape) # (10000,)
# print(y_test) # [7 2 1 ... 4 5 6]
# 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(len(y_test[0])) # 10
# print(y_test) # [[0. 0. 0. ... 1. 0. 0.] ...

x_train = (x_train/255).reshape(-1, 28*28)
x_test = (x_test/255).reshape(-1, 28*28)

x = np.append(x_train, x_test, axis=0)
print(x.shape)  # (70000, 784)

from sklearn.decomposition import PCA

pca = PCA(n_components=154)
pca.fit(x)
x = pca.transform(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)
# n_components = np.argmax(cumsum >= 0.95)+1  # argmax가 어찌 돌아가는지는 찍어봐야 알듯..
# # print(cumsum>=0.99) # True and False
# print(n_components) # 154

x_train = x[:60000,:]
x_test = x[60000:,:]

# 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Dense(100, input_dim=(154)))  # 아 그러면 첫번째 노드가 압축효과를 줬었겠구만..?
model.add(Dense(120))
# model.add(Dense())
# model.add(Dense(82))
model.add(Dense(80))
model.add(Dense(32))
model.add(Dense(10, activation='softmax')) # 확실히 수식을 보면서 해야 기억에서 안날라감

model.summary()

#3. 설명한 후 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 걍 loss만 바꿔주면 되네?
model.fit(x_train,y_train, epochs=200, batch_size=50)  # 훨낫네.....ㅎㅎ 와 근데 미세하게 계속 올라가긴 한다잉~

#4. 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
# print(predict)
print(np.argmax(predict, axis = 1))






