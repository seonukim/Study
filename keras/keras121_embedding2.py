# embedding

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
'''
첫번째 '너무' : index 1
11번째 '너무' : index 1

세번째 '참' : index 2
12번째 '참' : index 2

같은 단어는 동일 인덱스로 처리
-> 터미널 상에는 중복되어 출력되지 않는다

##################################################
'참'이 3개가 되었다
-> 참의 인덱스가 1이 됨
이를 통해 빈도수가 높은 단어의 인덱스 우선순위가 높음을 알 수 있음
'''

# 단어의 수치화 ; texts_to_sequences()
x = token.texts_to_sequences(docs)
print(x)
'''
return : 
[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23]]

인덱스의 조합으로 반환됨
하지만.. shape가 다른 문제가 발생 -> 해결해야한다
'''

# 원핫인코딩
# pad_sequences(): 와꾸를 맞추어, '의미'있는 숫자가 뒤로 가도록(padding = 'pre') 원핫인코딩 -> 0이 앞으로
# pad_sequences(): 와꾸를 맞추어, '의미'있는 숫자가 앞으로 가도록(padding = 'post') 원핫인코딩 -> 0이 뒤로
# value = 0이 아닌 다른 지정한 다른 숫자로 인코딩함
pad_x = keras.preprocessing.sequence.pad_sequences(x, padding = 'pre')
# pad_x = keras.preprocessing.sequence.pad_sequences(x, padding = 'post', value = 1.0)    # value = 1.0
'''
padding ; default = 'pre'
value   ; default = 0.
'''
print(pad_x)            # (12, 5)
'''여기까지 하면 데이터가 준비됨'''

## Embedding ; 원핫인코딩의 압축형

word_size = len(token.word_index) + 1
print(f'전체 토큰 사이즈 : {word_size}')                                # 전체 토큰 사이즈 : 25


## 모델링
model = keras.models.Sequential()
# model.add(keras.layers.Embedding(word_size, 10, input_length = 5))      # word_size = 전체 단어의 갯수
                                                                        # 10 : 다음 레이어의 output nodes, 두 번째 파라미터로 지정
                                                                        # input_length : 인풋 데이터의 columns
model.add(keras.layers.Embedding(20, 10, input_length = 5))             # word_size를 틀리게 입력한다고 해서 문제는 안되지만, 맞춰넣는게 가장 좋은 Accuracy
model.add(keras.layers.Flatten())                                       # Flatten으로 펼쳐준 후, Dense모델로 연결
model.add(keras.layers.Dense(1, activation = keras.activations.sigmoid))

model.summary()
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   

=================================================================
embedding_1 (Embedding)      (None, 5, 10)             250              Param # 250 : word_size * output_nodes

_________________________________________________________________
flatten_1 (Flatten)          (None, 50)                0

_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51        

=================================================================
Total params: 301
Trainable params: 301
Non-trainable params: 0
_________________________________________________________________
'''

## 컴파일
EPOCH = 30
model.compile(optimizer = keras.optimizers.Adam(lr = 1e-3),
              loss = keras.losses.binary_crossentropy,
              metrics = ['accuracy'])
model.fit(pad_x, labels, epochs = EPOCH)

acc = model.evaluate(pad_x, labels)[1]
print(f'Accuracy : {acc}')
