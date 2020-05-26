'''20200526 분류모델'''

# 회귀모델을 먼저 구현해보자.
# 1. 사전준비
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
from keras.losses import binary_crossentropy
import numpy as np

# 1-1. 조기종료 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

# 2. 데이터
x = np.array(range(1, 11))
y = np.array([1, 0 ,1, 0, 1, 0, 1, 0, 1, 0])
print(x.shape)
print(y.shape)

# 3. 모델 구성
model = Sequential()
model.add(Dense(10, input_shape = (1, )))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(1, activation = 'sigmoid'))
'''
분류모형의 활성화 함수 - 마지막 아웃풋 레이어에 추가
1. sigmoid
2. hard_sigmoid
3. softmax
'''
model.summary()

# 4. 실행 및 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x, y, epochs = 1000,
          batch_size = 1, callbacks = [es])
'''
분류모형의 손실함수
1. Cross-Entropy Loss
2. Categorical Cross-Entropy Loss
3. Binary Cross-Entropy Loss
4. Focal loss (함수로 구현해서 사용)
'''

# 5. 평가 및 예측
loss, acc = model.evaluate(x, y)
print("loss : ", loss)
print("acc : ", acc)

pred = model.predict(x)
print("pred : \n", pred)


# 기본적인 분류모델
# 결과치가 2가지로만 나오는 모델
# 이진 분류 모델 - Binary Classification Model
# activation 활성화 함수 - sigmoid
# 분류모형의 손실함수
# 1. Cross-Entropy Loss
# 2. Categorical Cross-Entropy Loss
# 3. Binary Cross-Entropy Loss
# 4. Focal loss (함수로 구현해서 사용)
# def focal_loss(gamma = 2., alpha = .25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
#                         - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#     return focal_loss_fixed

'''
pred :
 [[0.52978766]
  [0.5066741 ]
  [0.483532  ]
  [0.4604602 ]
  [0.43755686]
  [0.41491616]
  [0.39262962]
  [0.3707816 ]
  [0.34944952]
  [0.32870376]]     # 0.5를 기준으로 0, 1로 분류
'''