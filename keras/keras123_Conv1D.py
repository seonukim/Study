# 122를 copy해서 123에 paste
# embedding 없이 Conv1D로 모델 완성

import keras
import numpy as np
from keras.preprocessing.text import Tokenizer

# 영화평 리스트 생성
# docs = ['너무 재밌어요', '최고에요', '참 잘 만든 영화에요',
#         '추천하고 싶은 영화입니다', '한번 더 보고 싶네요',
#         '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요',
#         '재미없어요', '너무 재미없다', '참 재밌네요']

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요']

# 긍정표현 1, 부정표현 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])             # 모델의 최종 출력

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)        # fit_on_texts ; 단어 나열 인덱싱 형태로 반환
print(token.word_index)

# 단어의 수치화 ; texts_to_sequences()
x = token.texts_to_sequences(docs)
print(x)

# 원핫인코딩
pad_x = keras.preprocessing.sequence.pad_sequences(x, padding = 'pre')
print(pad_x)            # (12, 5)

pad_x = pad_x.reshape(12, 5, 1)
print(pad_x)            # (12, 5, 1)

word_size = len(token.word_index) + 1
print(f'전체 토큰 사이즈 : {word_size}')                                # 전체 토큰 사이즈 : 25


## 모델링
model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters = 10, kernel_size = 3,
                              input_shape = (5, 1), padding = 'same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, activation = keras.activations.sigmoid))

model.summary()


## 컴파일
EPOCH = 30
model.compile(optimizer = keras.optimizers.Adam(lr = 1e-3),
              loss = keras.losses.binary_crossentropy,
              metrics = [keras.metrics.binary_accuracy])
model.fit(pad_x, labels, epochs = EPOCH)

acc = model.evaluate(pad_x, labels)[1]
print(f'Accuracy : {acc}')          # Accuracy : 0.6666666865348816


'''
인풋 레이어부터 Conv1D로 구성하면
입력 데이터의 reshape 과정이 필요함
'''