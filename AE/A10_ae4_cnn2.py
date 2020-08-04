# a08_ae 복붙.
# cnn으로 오토인코더를 구성하시오

import tensorflow as tf

def autoencoder():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3),
                                     padding='valid', input_shape=(28,28,1),
                                     activation='relu')),
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3,3),
                                              padding='valid', activation='sigmoid')),
    return model


## 데이터
train_set, test_set = tf.keras.datasets.mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
x_train = x_train / 255.
x_test = x_test / 255.
print(x_train.shape)            # (60000, 28, 28, 1)
print(x_test.shape)             # (10000, 28, 28, 1)


model = autoencoder()
model.summary()

# model.compile(optimizer='adam', loss='mse', metrics=['acc'])  # loss = 0.01
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])    # loss = 0.09

model.fit(x_train, x_train, epochs = 10)

'''
output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set.yticks([])

plt.tight_layout()
plt.show()
'''