# 1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(219))
model.add(Dense(1))

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam
# optimizer = Adam(lr = 0.001)            # 옵티마이저 커스터마이징(튜닝)
                                          # Adam은 경사하강법 최적화 방법 중 하나이다
                                          # 클래스이므로 객체를 생성해서 사용 (이하 최적화함수 모두 동일)

# optimizer = RMSprop(lr = 0.001)         # loss: 0.0007303678430616856, 3.5032809
# optimizer = SGD(lr = 0.001)             # loss: 0.09778799116611481, 3.2912624
# optimizer = Adadelta(lr = 0.001)        # loss: 8.0286865234375, -0.12177757
# optimizer = Adagrad(lr = 0.001)         # loss: 0.7282077670097351, 2.36688
# optimizer = Adamax(lr = 0.001)          # loss: 0.04725462198257446, 3.3686242
# optimizer = Nadam(lr = 0.001)           # loss: 0.009725742973387241, 3.451579

model.compile(loss = 'mse',
              optimizer = optimizer,
              metrics = ['mse'])
model.fit(x, y, epochs = 100)

loss = model.evaluate(x, y)
print(f'loss: {loss[1]}')

pred1 = model.predict([3.5])
print(pred1)
'''
optimizer(최적화)의 종류와 파라미터
1) Adam : keras.optimizers.Adam(lr = 0.001,                 # 0보다 크거나 같은 float, 학습률
                                beta_1 = 0.9,               # 0보다 크고 1보다 작은 float, 일반적으로 1에 가깝게 설정됨
                                beta_2 = 0.999,             # 0보다 크고 1보다 작은 float, 일반적으로 1에 가깝게 설정됨
                                epsilon = None,             # 0보다 크거나 같은 float형 fuzz factor, None인 경우 K.epsilon()이 사용됨
                                decay = 0.0,                # 0보다 크거나 같은 float, 업데이트마다 적용되는 학습률의 감소율
                                amsgrad = False)            # boolean, Adam의 변형인 AMSGrad의 적용 여부 설정

2) RMSprop : keras.optimizers.RMSprop(lr = 0.001,           # 0보다 크거나 같은 float, 학습률
                                      rho = 0.9,            # 0보다 크거나 같은 float
                                      epsilon = None,       # 0보다 크거나 같은 float형 fuzz factor, None인 경우 K.epsilon()이 사용됨
                                      decay = 0.0)          # 0보다 크거나 같은 float, 업데이트마다 적용되는 학습률의 감소율

3) SGD : keras.optimizers.SGD(lr = 0.01,                    # 0보다 크거나 같은 float, 학습률
                              momentum = 0.0,               # 0보다 크거나 같은 float, SGD를 적절한 방향으로 가속화하며, 진동을 줄여줌
                              decay = 0.0,                  # 0보다 크거나 같은 float, 업데이트마다 적용되는 학습률의 감소율
                              nesterov = False)             # boolean, 네스테로프 모멘텀의 적용 여부 설정
# SGD는 Stochastic Gradient Descent, 확률적 경사하강법

4) Adadelta : keras.optimizers.Adadelta(lr = 1.0,           # 0보다 크거나 같은 float, 초기 학습률, default = 1, default 사용이 권장됨 
                                        rho = 0.95,         # 0보다 크거나 같은 float, 학습률 감소에 쓰이는 인자, 각 시점에 유지되는 경사하강의 비율
                                        epsilon = None,     # 0보다 크거나 같은 float형 fuzz factor, None인 경우 K.epsilon()이 사용됨
                                        decay = 0.0)        # 0보다 크거나 같은 float, 초기 학습률의 감소율
          