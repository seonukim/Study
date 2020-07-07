import keras
import numpy as np

## imdb 데이터 불러오기
imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = 10000)
'''
num_words : 훈련 데이터에서 가장 많이 등장하는 상위 10,000개 단어를 선택한다
'''

## 데이터 탐색
print(f'훈련 샘플 : {len(x_train)}, 레이블 : {len(y_train)}')   # 훈련 샘플 : 25000, 레이블 : 25000

## 첫번째 리뷰 확인해보기
print(x_train[0])
print(len(x_train[0]), len(x_train[1]))         # 218 189

## 정수를 단어로 다시 변환하기
# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()

# 처음 몇 개 인덱스는 사전에 정의되어 있음
word_index = {k:(v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2     # unknown
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(x_train[0]))
'''
<START>
this film was just brilliant casting location scenery story direction everyone's really suited
the part they played and you could just imagine being there robert <UNK> is an amazing actor
and now the same being director <UNK> father came from the same 
scottish island as myself so i loved the fact there was a real connection with this film the witty remarks
throughout the film were great it was just brilliant so much that i bought the film as soon as it was released
for <UNK> and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end
it was so sad and you know what they say if you cry at a film 
it must have been good and this definitely was also <UNK> to the 
two little boy's that played the <UNK> of norman and paul they were just brilliant
children are often left out of the <UNK> list i think because the stars that play them all grown up
are such a big profile for the whole film but these children are amazing and 
should be praised for what they have done don't you think the whole story was so lovely
because it was true and was someone's life after all that was shared with us all
'''

## 데이터 준비
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train, value = 0.,
    padding = 'pre', maxlen = 256)
x_test = keras.preprocessing.sequence.pad_sequences(
    x_test, value = 0.,
    padding = 'pre', maxlen = 256)
print(len(x_train[0]))              # 256
print(len(x_train[1]))              # 256

# 첫번째 리뷰 전체 확인해보기
print(x_train[0])

#### imdb 데이터셋은 이진분류이기 때문에(긍정, 부정) 레이블 원핫인코딩은 할 필요가 없다 !!

## 모델링
model = keras.models.Sequential([
    keras.layers.Embedding(input_dim = 21904, output_dim = 256),
    keras.layers.Dropout(rate = 0.3),
    # keras.layers.Bidirectional(keras.layers.LSTM(units = 10)),
    keras.layers.LSTM(units = 10),              # params 갯수가 2배로 증가함
    keras.layers.BatchNormalization(),
    keras.layers.Activation('elu'),
    keras.layers.Dropout(rate = 0.3),
    keras.layers.Dense(units = 64),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('elu'),
    keras.layers.Dropout(rate = 0.3),
    keras.layers.Dense(units = 32),
    keras.layers.Dense(units = 1),
    keras.layers.Activation('sigmoid')
])

model.summary()

'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   

=================================================================
embedding_1 (Embedding)      (None, None, 256)         2560000   

_________________________________________________________________
dropout_1 (Dropout)          (None, None, 256)         0

_________________________________________________________________
lstm_1 (LSTM)                (None, 128)               197120    

_________________________________________________________________
batch_normalization_1 (Batch (None, 128)               512       

_________________________________________________________________
activation_1 (Activation)    (None, 128)               0

_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0

_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      

_________________________________________________________________
batch_normalization_2 (Batch (None, 64)                256       

_________________________________________________________________
activation_2 (Activation)    (None, 64)                0

_________________________________________________________________
dropout_3 (Dropout)          (None, 64)                0

_________________________________________________________________
dense_2 (Dense)              (None, 32)                2080      

_________________________________________________________________
dense_3 (Dense)              (None, 1)                 33        

_________________________________________________________________
activation_3 (Activation)    (None, 1)                 0

=================================================================
Total params: 2,768,257
Trainable params: 2,767,873
Non-trainable params: 384
_________________________________________________________________
'''
'''
## 컴파일 및 훈련
model.compile(loss = keras.losses.binary_crossentropy,
              optimizer = keras.optimizers.Adam(lr = 1e-4),
              metrics = [keras.metrics.binary_accuracy])
hist = model.fit(x_train, y_train,
                 epochs = 20, batch_size = 100,
                 validation_split = 0.2, verbose = 1)
print(hist.history.keys())

## 모델 평가
res = model.evaluate(x_test, y_test)
print(f'Loss : {res[0]}')
print(f'Acc : {res[1]}')'''