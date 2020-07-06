# NLP ; Embedding

import keras
from keras.preprocessing.text import Tokenizer

text = '나는 맛있는 밥을 먹었다'

'''자연어 처리의 기본'''
token = Tokenizer()
token.fit_on_texts([text])              # 토큰화: 지정한 문장을 '단어'단위로 자른 후 딕셔너리화한다

print(token.word_index)                 # {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}

x = token.texts_to_sequences([text])    # 문자열의 인덱스만 반환
print(x)                                # [[1, 2, 3, 4]]

# 문자열 인덱싱에 가치는 없다
# 카레가 5, 오므라이스가 1이면, 카레는 오므라이스의 5배의 가치? -> x
# 따라서, 범주화를 시켜줘야 한다
# 방법은? 원핫인코딩
word_size = len(token.word_index) + 1       # to_categorical의 문제점: 첫 열의 인덱스가 0부터 시작, 따라서 + 1
x = keras.np_utils.to_categorical(x, num_classes = word_size)
print(x)

# 문제점: 열이 많아진다 -> 데이터가 많아진다.
# 데이터가 커지면? -> 압축..
# 압축을 위해 등장한 개념: Embedding -> 자연어처리, 시계열에서 많이 사용함
