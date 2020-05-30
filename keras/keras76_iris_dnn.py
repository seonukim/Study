import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
rs = RobustScaler()
mms = MinMaxScaler()
ss = StandardScaler()
mas = MaxAbsScaler()
pca = PCA(n_components = 2)


''' 1. load data '''
x, y = load_iris(return_X_y = True)
print(x.shape)                  # (150, 4)
print(y.shape)                  # (150,)

# 1-1. preprocessing
pca.fit(x)
x = pca.transform(x)

# 1-2. normalization
x = mas.fit_transform(x)

# 1-3. data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

print(x_train.shape)            # (120, 4)
print(x_test.shape)             # (30, 4)
print(y_train.shape)            # (120,)
print(y_test.shape)             # (30,)

# 1-4. One Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train[0])
print(y_test[0])


''' 2. Modeling _ DNN '''

# 2-1. Sequential Model
model = Sequential()

model.add(Dense(128, input_shape = (2, ),
                activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(3, activation = 'softmax'))

'''
# 2-2. Function Model
input1 = Input(shape = (4, ))
layer1 = Dense(128, activation = 'relu')(input1)
layer2 = Dense(128, activation = 'relu')(layer1)
layer3 = Dropout(rate = 0.2)(layer2)
layer4 = Dense(256, activation = 'relu')(layer3)
layer5 = Dense(256, activation = 'relu')(layer4)
layer6 = Dropout(rate = 0.2)(layer5)

output1 = Dense(512, activation = 'relu')(layer6)
output2 = Dense(512, activation = 'relu')(output1)
output3 = Dropout(rate = 0.2)(output2)
output4 = Dense(1, activation = 'sigmoid')(output3)

model = Model(inputs = input1, outputs = output4)
'''
model.summary()


''' 3. compile & fitting '''
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 1000, batch_size = 1,
                 validation_split = 0.05, verbose = 1)

print(hist.history.keys())


''' 4. Evaluate Model '''
res = model.evaluate(x_test, y_test)
print("=" * 35)
print("Result : ", res)
print("loss : ", res[0])
print("acc : ", res[1])

pred = model.predict(x_test)
pred = np.argmax(pred, axis = 1)
print("=" * 35)
print("Predict of 1 ~ 5 : \n", pred[:5])

y_test = np.argmax(y_test, axis = 1)
print("Test data of 1 ~ 5 : \n", y_test[:5])


'''
Result 1)
- PCA, Scaler 미적용
loss :  0.2529725134372711
acc  :  0.8444444537162781
===================================
Predict of 1 ~ 5 : 
 [2 2 2 1 0]


 Result 2)
 - PCA = 2, Scaler 미적용
loss :  0.7056805491447449
acc  :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 3)
 - PCA = 2, MinMaxScaler 적용
loss :  0.35385245084762573
acc  :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 4)
 - PCA = 2, StandardScaler 적용
loss :  0.653983473777771
acc  :  0.9333333373069763
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 5)
 - PCA = 2, RobustScaler 적용
loss :  0.4314327836036682
acc  :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 6)
 - PCA = 2, RobustScaler 적용
loss :  0.23068833351135254
acc :  0.9777777791023254
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 7)
 - PCA 미적용, StandardScaler 적용
loss :  0.6800507307052612
acc :  0.9111111164093018
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 8)
 - PCA 미적용, MinMaxScaler 적용
loss :  0.07252607494592667
acc :  0.9777777791023254
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 9)
 - PCA 미적용, RobustScaler 적용
loss :  10.568984985351562
acc :  0.9111111164093018
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 10)
 - PCA 미적용, MaxAbsScaler 적용
loss :  0.10713689774274826
acc :  0.9333333969116211
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]
'''

plt.figure(figsize = (10, 6))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.',
         c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.',
         c = 'blue', label = 'val_loss')
plt.title("loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['accuracy'], marker = '.',
         c = 'violet', label = 'acc')
plt.plot(hist.history['val_accuracy'], marker = '.',
         c = 'green', label = 'val_acc')
plt.title('accuracy')
plt.ylim(0, 1.0)
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc = 'lower right')

plt.show()