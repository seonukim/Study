# 1. 필요한 모듈 임포트
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 2. 데이터 준비
# 데이터는 훈련용 데이터와 평가용 데이터, 예측용 데이터로 준비한다.
x_train = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
y_train = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
x_test = np.array([110, 220, 330, 440, 550, 660, 770, 880, 990, 1100])
y_test = np.array([110, 220, 330, 440, 550, 660, 770, 880, 990, 1100])
x_pred = np.array([1000, 2000, 3000, 4000, 5000])

# 3. 모델 구성 - DNN(Deep Neural Network)
model = Sequential()
model.add(Dense(10, input_dim = 1, activation = 'relu'))
model.add(Dense(187))
model.add(Dense(148))
model.add(Dense(139))
model.add(Dense(193))
model.add(Dense(148))
model.add(Dense(153))
model.add(Dense(142))
model.add(Dense(126))
model.add(Dense(198))
model.add(Dense(1, activation = 'relu'))

# 4. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, batch_size = 1, epochs = 100)

# 5. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse : ", mse)

# 6. 예측값 출력
y_pred = model.predict(x_pred)
print("y의 예측값은? ", y_pred)


"""
* 내가 구성한 모델의 결과값 비교하기 *

! Result 1 - 초기 모델
loss :  0.055168033484369515
 mse :  0.0551680326461792
y의 예측값은?  [[ 999.6247]
               [1999.0721]
               [2998.5208]
               [3997.9688]
               [4997.4165]]


! Result 2 - epochs 조절 ; 반복횟수 감소 100 -> 80
loss :  0.003411769587546587
 mse :  0.003411769401282072
y의 예측값은?  [[ 999.94006]
               [1999.7588 ]
               [2999.5776 ]
               [3999.395  ]
               [4999.213  ]]
-> epochs를 감소 시키는 모델링의 결과 손실률이 낮아짐(좀 더 좋은 모델)을 확인함


! Result 3 - epochs 조절 ; 반복횟수 증가 80 -> 100
             layers 조절 ; hidden layer 늘리기 5 -> 10
loss :  1201.7285179138185
 mse :  1201.728515625
y의 예측값은?  [[ 948.9153]
               [1896.8297]
               [2844.7444]
               [3792.6597]
               [4740.574 ]]
-> 최악의 결과가 출력됨. loss가 1000을 넘어가는 사태 발생


! Result 4 - epochs 조절 ; 횟수 증가 100 -> 500
             batch_size 조절 ; 1 -> 2
loss :  0.08594375029206276
 mse :  0.08594374358654022
y의 예측값은?  [[ 999.55664]
               [1998.6819 ]
               [2997.8066 ]
               [3996.933  ]
               [4996.0576 ]]
-> batch_size를 1에서 2로 늘렸더니 loss가 많이 감소(좀 더 좋은 모델)됨을 확인함
-> epochs도 100 -> 500으로 대폭 늘렸다.


! Result 5 - hidden layer의 nodes 갯수 조절 ; 일괄적으로 각 layer당 node * 10
             epochs를 다시 100으로 되돌림
             batch_size를 다시 1로 되돌림
loss :  465850.0
 mse :  465850.0
y의 예측값은?  [[0.]
               [0.]
               [0.]
               [0.]
               [0.]]
-> 해당 모델은 완전히 실패한 모델이다.


! Result 6 - hidden layer의 nodes 갯수 조절 ; 일괄적으로 줄임
             이외의 hyper-parameter는 불변
loss :  0.35923462323844435
 mse :  0.35923463106155396
y의 예측값은?  [[ 999.121 ]
               [1997.2887]
               [2995.456 ]
               [3993.6243]
               [4991.794 ]]
-> loss는 감소했지만 출력된 예측값이 애매하다.

"""