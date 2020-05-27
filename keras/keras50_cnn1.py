'''합성곱 신경망, CNN ; Convolution Neural Network'''

from keras.models import Sequential
from keras.layers import Conv2D         # Convolution; 합성곱
from keras.layers import MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape = (10, 10, 1)))                # (9, 9, 10)
'''
10 : 첫 번째 레이어의 아웃풋
(2, 2) : 가로, 세로
input_shape = (10, 10, 1) ; 가로, 세로, 흑백여부(1 or 3, 3 : color), 행, 열, 채널 수(흑백 영상 = 1, RGB = 3)
x = (10000, 10, 10, 1)
     행    가로 세로 명암(특성)        -> CNN : 4차원, 장수는 무시!
'''
model.add(Conv2D(7, (3, 3)))                                            # (7, 7, 7)
model.add(Conv2D(5, (2, 2), padding = 'same'))                          # (7, 7, 5)
model.add(Conv2D(5, (2, 2)))                                            # (6, 6, 5)
# model.add(Conv2D(5, (2, 2), strides = 2))
# model.add(Conv2D(5, (2, 2), strides = 2, padding = 'same'))             # (3, 3, 5)
model.add(MaxPooling2D(pool_size = 2))                                  # 취향, 선택
model.add(Flatten())                                                    # 이전 레이어의 아웃풋을 다 곱해서 2차원으로 차원을 축소함
model.add(Dense(1))

model.summary()

'''
Conv2D의 파라미터
위의 예시 모델에서,
1) 10 ; filters = 출력 공간의 치수(즉, convolution의 출력 필터 수)
2) (2, 2) ; kernel_size = 2D convolution 창의 높이와 너비, 모든 공간적 차원
3) input_shape ; 입력 레이어의 차원 = 3 채널 10 x 10 이미지에 대한 인풋 텐서, 샘플 수를 제외한 입력 형태 정의
    - (batch_size, height, width, channels)
4) strides = 높이와 너비를 따라 convolution의 폭을 명시함, 연산을 수행할 때 윈도우가 가로 세로로 움직이면서
             내적 연산을 수행하는데, 한 번에 얼마나 움직일지를 결정함(default = 1)
5) padding ; 경계 처리 방법
    - 'valid' : 유효한 영역만 출력, 출력 이미지 사이즈는 입력 사이즈보다 작음(default)
    - 'same' : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일함
6) 우선순위 : strides > padding
7) num of params = (input x kernel x kernel + bias) x output
'''