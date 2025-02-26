# A11_noise1.py를 copy
# A12_noise2_cnn.py에 paste

import tensorflow as tf
import numpy as np

# 모델 정의하기
def autoencoder():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),
                               input_shape=(28,28,1), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3,3), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation='sigmoid')
    ])
    return model

## 데이터
train_set, test_set = tf.keras.datasets.mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
x_train = x_train / 255.
x_test = x_test / 255.

# 입력 데이터에 노이즈 생성 ; 픽셀에 랜덤하게 0을 뿌려준다
# 문제가 있다 ; / 255.를 해줌으로써 현재 데이터의 분포는 0 ~ 1 사이에 있음
# 평균이 0, 표준편차가 0.5면 0 ~ 1 사이의 범위를 벗어날 수 있음
x_train_noised = x_train + np.random.normal(0, 0.01, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.01, size=x_test.shape)
                                # random.normal : 정규분포에 의한 난수 생성
                                # 0   : 평균
                                # 0.5 : 표준편차
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
# np.clip(array, min, max)
# array 내의 element들에 대해서
# min값보다 작은 값들을 min값으로 바꿔주고
# max값보다 큰 값들을 max값으로 바꿔주는 함수
# 배열의 범위를 제한한다

model = autoencoder()

model.summary()
print("=" * 40)
print(x_train_noised.shape)         # (60000, 28, 28, 1)
print(x_test_noised.shape)          # (10000, 28, 28, 1)

# model.compile(optimizer='adam', loss='mse', metrics=['acc'])  # loss = 0.01
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])    # loss = 0.0836

model.fit(x_train_noised, x_train, epochs=10, batch_size=128)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5),
      (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
          plt.subplots(3, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다            ; subplot의 첫 번째 행
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 이미지를 두 번째 줄에 그린다           ; subplot의 두 번째 행
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다    ; subplot의 세 번째 행
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# plt.tight_layout()
plt.show()