import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
# MacOS
path = '/Users/seonwoo/Desktop/Dacon/'
os.chdir(path)

# 10.1.3 퍼셉트론
iris = load_iris()
x = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron()
per_clf.fit(x, y)

y_pred = per_clf.predict([[2, 0.2]])
print(y_pred)                   # [1]


# 10.2.1 텐서플로2 설치
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)           # 2.1.0
print(keras.__version__)        # 2.2.4-tf


# 10.2.2 시퀀셜 API를 사용하여 이미지 분류기 만들기
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)       # (60000, 28, 28)
print(X_train_full.dtype)       # uint8

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(class_names[y_train[0]])      # Coat

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
print(model.layers)
'''
[<tensorflow.python.keras.layers.core.Flatten object at 0x00000213366BD808>,
 <tensorflow.python.keras.layers.core.Dense object at 0x00000213366B1948>,
 <tensorflow.python.keras.layers.core.Dense object at 0x00000213366B1FC8>,
 <tensorflow.python.keras.layers.core.Dense object at 0x00000213366C7248>]
'''

hidden1 = model.layers[1]
print(hidden1.name)         # dense_3
print(model.get_layer(hidden1.name) is hidden1)        # True

weights, biases = hidden1.get_weights()
print(weights)
print(weights.shape)        # (784, 300)
print(biases)
print(biases.shape)         # (300,)

model.compile(loss = keras.losses.sparse_categorical_crossentropy,
              optimizer = keras.optimizers.SGD(),
              metrics = [keras.metrics.sparse_categorical_accuracy])

history = model.fit(X_train, y_train, epochs = 30,
                    validation_data = (X_valid, y_valid))
print(history.params)
'''
{'batch_size': 32, 'epochs': 30, 'steps': 1719,
 'samples': 55000, 'verbose': 0, 'do_validation': True,
 'metrics': ['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy']}
'''
print(history.epoch)
'''
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
 21, 22, 23, 24, 25, 26, 27, 28, 29]
'''
print(history.history.keys())
'''
dict_keys(['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy'])
'''

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
# plt.show()

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
print(y_pred)           # [9 2 1]

np.array(class_names)[y_pred]
print(np.array(class_names)[y_pred])    # ['Ankle boot' 'Pullover' 'Trouser']

y_new = y_test[:3]
print(y_new)            # [9 2 1]


# 10.2.3 시퀀셜 API를 사용하여 회귀용 다층 퍼셉트론 만들기
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state = 42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation = "relu", input_shape = X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss = "mean_squared_error", optimizer = 'sgd')
history = model.fit(X_train, y_train, epochs = 20,
                    validation_data = (X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
print(y_pred)
'''
[[0.6800924]
 [1.7144277]
 [4.1082516]]
'''


# 10.2.4 함수형 API를 사용해 복잡한 모델 만들기
input_ = keras.layers.Input(shape = X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation = "relu")(input_)
hidden2 = keras.layers.Dense(30, activation = "relu")(hidden1)
concat = keras.layers.concatenate([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs = [input_], outputs = [output])

input_A = keras.layers.Input(shape = [5], name = "wide_input")
input_B = keras.layers.Input(shape = [6], name = "deep_input")
hidden1 = keras.layers.Dense(30, activation = "relu")(input_B)
hidden2 = keras.layers.Dense(30, activation = "relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name = "output")(concat)
model = keras.models.Model(inputs = [input_A, input_B], outputs = [output])

model.compile(loss = "mse", optimizer = keras.optimizers.SGD(lr = 1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs = 20,
                    validation_data = ((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))
print(y_pred)
'''
[[0.5567778]
 [1.9675388]
 [3.2182348]]
'''

# 규제를 위해 보조 출력 추가
input_A = keras.layers.Input(shape = [5], name = "wide_input")
input_B = keras.layers.Input(shape = [6], name = "deep_input")
hidden1 = keras.layers.Dense(30, activation = "relu")(input_B)
hidden2 = keras.layers.Dense(30, activation = "relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
# 보조 출력(위는 이전과 동일)
output = keras.layers.Dense(1, name = "main_output")(concat)
aux_output = keras.layers.Dense(1, name = "aux_output")(hidden2)
model = keras.models.Model(inputs = [input_A, input_B],
                           outputs = [output, aux_output])

model.compile(loss = ["mse", "mse"],
              loss_weights = [0.9, 0.1],
              optimizer = 'sgd')

history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs = 20,
                    validation_data = ([X_valid_A, X_valid_B], [y_valid, y_valid]))

# 모델 평가 -> 개별 손실과 총 손실을 return
total_loss, main_loss, aux_loss = model.evaluate(
    [X_test_A, X_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
print(y_pred_main)
print(y_pred_aux)
'''
[[0.5748284]
 [1.7350574]
 [3.4623878]]
[[0.69406366]
 [1.9380082 ]
 [3.0294228 ]]
'''


# 10.2.5 서브클래싱 API로 동적 모델 만들기
class WideAndDeepModel(keras.models.Model):
    def __init__(self, units = 30, activation = "relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation = activation)
        self.hidden2 = keras.layers.Dense(units, activation = activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
        
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()


# 10.2.6 모델 저장과 복원
model = keras.models.Sequential([
    keras.layers.Dense(30, activation = "relu", input_shape = [8]),
    keras.layers.Dense(30, activation = "relu"),
    keras.layers.Dense(1)
])

model.compile(loss = "mse", optimizer = 'sgd')
history = model.fit(X_train, y_train, epochs = 10, validation_data = (X_valid, y_valid))

# 모델 저장
model.save(path + 'my_keras_model.h5')

# 모델 로드
model = keras.models.load_model(path + 'my_keras_model.h5')


# 10.2.7 콜백 사용하기
cp = keras.callbacks.ModelCheckpoint(filepath = path + 'my_keras_model.h5',
                                     save_best_only = True)
history = model.fit(X_train, y_train, epochs = 10,
                    validation_data = (X_valid, y_valid),
                    callbacks = [cp])
model = keras.models.load_model(path + 'my_keras_model.h5')

# 조기 종료; EarlyStopping 구현
es = keras.callbacks.EarlyStopping(patience = 10,
                                   restore_best_weights = True)
history = model.fit(X_train, y_train, epochs = 100,
                    validation_data = (X_valid, y_valid),
                    callbacks = [cp, es])

# 사용자 정의 콜백; 다음 콜백은 훈련하는 동안 검증 손실과 훈련 손실의 비율을 출력함
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


# 10.2.8 텐서보드를 사용해 시각화하기
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
print(run_logdir)

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs = 30,
                    validation_data = (X_valid, y_valid),
                    callbacks = [cp, tensorboard_cb])

# 텐서플로는 tf.summary패키지로 저수준 API를 제공함
# 다음 코드는 create_file, writer() 함수를 사용해 SummaryWriter를 만들고
# with 블럭 안에서 텐서보드를 사용해 시각화 할 수 있는 스칼라, 히스토그램,
# 이미지, 오디오, 텍스트를 기록한다
'''
test_logdir = get_run_logdir
writer = tf.summary.create_file_writer(test_logdir)
with writer.as_default():
    for step in range(1, 1000 + 1):
        tf.summary.scalar('my_scalar', np.sin(step / 10), step = step)
        data = (np.random.randn(100) + 2) * step / 100
        tf.summary.scalar('my_hist', data, buckets = 50, step = step)
        images = np.random.rand(2, 32, 32, 3)       # 32*32 RGB이미지
        tf.summary.image('my_images', images * step / 1000, step = step)
        texts = ['The step is ' + str(step), 'Its square is ' + str(step **2)]
        tf.summary.text('my_text', texts, step = step)
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio('my_audio', audio, sample_rate = 48000, step = step)
'''


# 10.3 신경망 하이퍼파라미터 튜닝하기
def build_model(n_hidden = 1, n_neurons = 30, learning_rate = 3e-3, input_shape = [8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation = "relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr = learning_rate)
    model.compile(loss = "mse", optimizer = optimizer)
    return model

# Scikit_learn 래핑
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
keras_reg.fit(X_train, y_train, epochs = 100,
              validation_data = (X_valid, y_valid),
              callbacks = [keras.callbacks.EarlyStopping(patience = 10)])
mse_test = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)
print(y_pred)

# RandomizedSearchCV
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs,
                                   n_iter = 10, cv = 3, verbose = 2)
rnd_search_cv.fit(X_train, y_train, epochs = 100,
                  validation_data = (X_valid, y_valid),
                  callbacks = [es])
print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)
model = rnd_search_cv.best_estimator_.model

#####################################################################
'''11장 심층 신경망 훈련하기 내용 추가'''

## 활성화 함수
# He 정규분포 초기화
model = keras.models.Sequential([
    keras.layers.Dense(30, activation = "relu", input_shape = [8]),
    keras.layers.Dense(30, activation = "relu"),
    keras.layers.Dense(10, kernel_initializer = 'he_normal'),
    keras.layers.LeakyReLU(alpha = 0.2),
    keras.layers.Dense(1)
])

# 르쿤 정규분포 초기화
layer = keras.layers.Dense(10, activation = 'selu',
                           kernel_initializer = 'lecun_normal')

# 배치정규화
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation = 'elu',
                       kernel_initializer = 'he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation = 'elu',
                       kernel_initializer = 'he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation = 'softmax')
])

model.summary()

# 첫 번째 배치 정규화 층의 파라미터 살펴보기
print([(var.name, var.trainable) for var in model.layers[1].variables])
print(model.layers[1].updates)

# 활성화 함수 전에 배치 정규화 층 추가하기
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer = 'he_normal', use_bias = False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('elu'),
    keras.layers.Dense(100, kernel_initializer = 'he_normal', use_bias = False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('elu'),
    keras.layers.Dense(10, activation = 'softmax')
])

## 그래디언트 클리핑
# RNN 등의 순환신경망에서는 배치정규화를 적용하기 어렵다
# 따라서 그래디언트 폭주 문제를 완화하기 위해,
# 그래디언트가 역전파될 때 일정 임곗값을 넘어서지 못하도록 그래디언트를 잘라낸다
# optimizer 변수를 만들 때 clipvalue | clipnorm 매개변수를 지정한다
optimizer = keras.optimizers.SGD(clipvalue = 1.0)
model.compile(loss = 'mse', optimizer = optimizer)

## 전이학습; 사전훈련된 층 재사용하기
'''실행은 하지 않고 코드만 작성한다'''
model_A = keras.models.load_model('불러올 모델')
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation = 'sigmoid'))

# model_A와 model_B_on_A는 일부 층을 공유한다, model_B_on_A를 훈련할 때 model_A도 영향을 받음
# 이를 원치 않는다면 층을 재사용하기 전에 model_A를 clone(복제)한다.
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())        # 가중치 가져오기

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss = 'binary_crossentropy',
                     optimizer = 'sgd',
                     metrics = ['accuracy'])

history = model_B_on_A.fit(X_train_B, y_train_B, epochs = 4,
                           validation_data = (X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(lr = 1e-4)     # default = 1e-2
model_B_on_A.compile(loss = 'binary_crossentropy',
                     optimizer = optimizer,
                     metrics = ['accuracy'])
histrory = model_B_on_A.fit(X_train_B, y_train_B, epochs = 16,
                            validation_data = (X_valid_B, y_valid_B))
model_B_on_A.evaluate(X_test_B, y_test_B)

# 고속 옵티마이저
# 모멘텀 최적화
optimizer = keras.optimizers.SGD(lr = 1e-3, momentum = 0.9)

# 네스테로프 가속 경사(NAG)
optimizer = keras.optimizers.SGD(lr = 1e-3, momentum = 0.9, nesterov = True)

# RMSprop 최적화
optimizer = keras.optimizers.RMSprop(lr = 1e-3, rho = 0.9)  # rho는 감쇠율 베타

# Adam 최적화
optimizer = keras.optimizers.Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999)
'''
beta_1 = 모멘텀 감쇠 하이퍼파라미터(default = 0.9)
beta_2 = 스케일 감쇠 하이퍼파라미터(default = 0.999)
'''

# 학습률 스케쥴링
'''
s = 학습률을 나누기 위해 수행할 스텝 수
keras는 c를 1로 가정한다
'''
# 거듭제곱 기반 스케줄링
optimizer = keras.optimizers.SGD(lr = 1e-3, decay = 1e-4)   # decay는 s의 역수

# 지수기반 스케줄링
# 현재 에포크를 받아 학습률을 반환하는 함수
def exponential_decay_fn(epoch):
    return 0.01 * 0.1**(epoch / 20)

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0 = 0.01, s = 20)

# 그 다음, 이 스케쥴링 함수를 전달하여 LearningRateScheduler 콜백을 만든다
# 그리고 이 콜백을 fit()에 전달
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, callbacks = [lr_scheduler])

# 두 번째 매개변수로 현재 학습률을 받는 스케쥴 함수
def exponential_decay_fn(epoch, lr):
    return lr * 0.1**(1 / 20)

# 구간별 고정 스케쥴링
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
# 성능 기반 스케쥴링을 위해 ReduceLROnPlateau 콜백 사용
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor = 0.5, patience = 5)

# 성능 기반 스케쥴링
'''1사이클 스케쥴링'''  # <- 이해되지 않으므로 나중에 다시 읽어보기
'''p.445'''

# 에포크가 아니라 매 스텝마다 학습률 업데이트하기
# 앞서 정의한 exponential_decay_fn과 동일한 지수 기반 스케쥴링을 구현
s = 20 * len(X_train) // 32     # 20번 에포크에 담긴 전체 스텝 수(batch_size = 32)
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)

## 규제를 사용해 과대적합 피하기
# l2규제
layer = keras.layes.Dense(100, activation = 'elu',
                          kernel_initializer = 'he_normal',
                          kernel_regularizer = keras.regularizers.l2(0.01))

# l1, l2 규제 모두 사용하기
layer = keras.layes.Dense(100, activation = 'elu',
                          kernel_initializer = 'he_normal',
                          kernel_regularizer = keras.regularizers.l1_l2(0.01, 0.01))

# python의 functools.partial()함수를 사용하여 기본 매개변수 값을 사용하여 함수 호출을 감싼다
from functools import partial
RegularizedDense = partial(keras.layers.Dense,
                           activation = 'elu',
                           kernel_initializer = 'he_normal',
                           kernel_regularizer = keras.regularizers.l2(0.01))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation = 'softmax',
                     kernel_initializer = 'glorot_uniform'),
])

# 드롭아웃 규제 사용하기
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dropout(rate = 0.2),
    keras.layers.Dense(300, activation = 'elu',
                       kernel_initializer = 'he_normal'),
    keras.layers.Dropout(rate = 0.2),
    keras.layers.Dense(100, activatio = 'elu',
                       kernel_initializer = 'he_normal'),
    keras.layers.Dropout(rate = 0.2),
    keras.layers.Dense(10, activation = 'softmax')
])

# 몬테 카를로 드롭아웃
# MC 드롭아웃; 모델의 불확실성을 더 잘 측정할 수 있다
y_probas = np.stack([model(X_test_scaled, training = True)
                     for sample in range(100)])
y_proba = y_probas.mean(axis = 0)

# 드롭아웃을 비활성화한 패션mnist 테스트 세트에 있는 첫 번째 샘플 모델 예측
np.round(model.predict(X_test_scaled[:1]), 2)

# 드롭아웃을 활성화하여 만든 예측
np.round(y_probas[:1], 2)

# 첫 번째 차원으로 평균을 낸 MC 드롭아웃의 예측
np.round(y_proba[:1], 2)

# 위의 확률 추정의 표준 분포 확인
y_std = y_probas.std(axis = 0)
np.round(y_std[:1], 2)

# 정확도 확인
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(accuracy)         # 0.8694

# Dropout층을 MCDropout 클래스로 바꾸기
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training = True)

# Max_norm regularizer
# 맥스-노름 규제
# 맥스-노름의 하이퍼파라미터인 r을 줄이면 규제의 양이 증가하여 과적합을 감소시키는데 도움이 된다
# 불안정한 그래디언트 문제를 완화하는데 도움이 됨
keras.layers.Dense(100, activation = 'elu',
                   kernel_initializer = 'he_normal',
                   kernel_constraint = keras.constraint.max_norm(1.))
