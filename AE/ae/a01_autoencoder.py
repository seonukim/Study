'''
acc = 98% 이상 !
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# # 전처리
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

x_train = (x_train/255).reshape(-1, 28*28)
x_test = (x_test/255).reshape(-1, 28*28)  # 0부터 1사이 값으로 변하게 되어서 나중에 sigmoid 씌워줌

# 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input

input = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2)

encoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(encoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


'''
#3. 설명한 후 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 걍 loss만 바꿔주면 되네?
model.fit(x_train,y_train, epochs=200, batch_size=900)  # 훨낫네.....ㅎㅎ 와 근데 미세하게 계속 올라가긴 한다잉~

#4. 평가와 예측
loss, acc = model.evaluate(x_test,y_test) 
print('loss 는',loss)
print('acc 는',acc)

predict = model.predict(x_test)
# print(predict)
print(np.argmax(predict, axis = 1))


# 아 분류에서 사실상 rmse, r2 볼 필요가 없는거 같은데?
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(predict, y_test) )

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(predict, y_test)
print('R2는 ', r2)
'''




