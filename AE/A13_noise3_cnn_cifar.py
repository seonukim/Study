# cifar10으로 autoencoder를 구성할 것

import tensorflow as tf
import matplotlib.pyplot as plt

## 모델 정의
def autoencoder():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3),
                               padding='valid', input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3,3), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation='sigmoid')
    ])
    return model

## 데이터 로드
(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.cifar10.load_data()
print(x_train.shape)            # (50000, 32, 32, 3)
print(x_test.shape)             # (10000, 32, 32, 3)

## 전처리
x_train = x_train / 255.
x_test = x_test / 255.

## 모델 구성
model = autoencoder()
model.summary()

## 컴파일 훈련
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, x_train, epochs=3, batch_size=128)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5),
      (ax6, ax7, ax8, ax9, ax10)) = \
          plt.subplots(2, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다            ; subplot의 첫 번째 행
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(32, 32, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다    ; subplot의 2 번째 행
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(32, 32, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()