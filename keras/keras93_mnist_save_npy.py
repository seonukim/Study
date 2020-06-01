# from keras93

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

# load proprcessed data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])  # image. 0~255 
# print(f"y_train[0] : {y_train[0]}")

print(x_train.shape)     # (60000, 28, 28) 60000 28*28 images
print(x_test.shape)      # (10000, 28, 28) 10000 28*28 images  
print(y_train.shape)     # (60000,)  60000 scalars
print(y_test.shape)      # (10000,)  10000 scalars

# save npy file b4 preprocessing
np.save('./data/mnist_train_x.npy', arr=x_train)
np.save('./data/mnist_test_x.npy', arr=x_test)
np.save('./data/mnist_train_y.npy', arr=y_train)
np.save('./data/mnist_test_y.npy', arr=y_test)

'''
# plt.imshow(x_train[33123], 'inferno_r')  # imshow == show images
# plt.imshow(x_train[0])  
# plt.show()
# data preprocessing 1. one_hot_encoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)  # always check the shape!!
# data preprocessing 2. normalization
x_train = x_train.reshape(60000, 28, 28, 1)/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.
# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255 # or /255. , /255.0
# x_test = x_test.reshape(60000, 28, 28, 1).astype('float32')/255
# compile, fit
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(28,28,1), activation='relu')) 
model.add(Dropout(rate=0.2))
model.add(Conv2D(64, (3,3), activation='relu')) 
model.add(Dropout(rate=0.3))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(10, activation='softmax'))
# model.save(filepath='.\model\keras85_model\model_test01.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
model_filepath='.\model\keras90_model\{epoch:02d}-{val_loss:.4f}.hdf5'
model_checkpoint = ModelCheckpoint(filepath=model_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
# .fit returns history
hist = model.fit(x_train, y_train, batch_size=1024, epochs =10, verbose=1, callbacks=[early_stopping, model_checkpoint],  validation_split = 0.1)
model = load_model('.\\model\\keras90_model\\07-0.0382.hdf5')  # \ 사용시에 퍼미션에러가 뜸, 다양한 방법으로 시도 바람. (\\, //, /, \)
# evaluate, predict
loss_acc = model.evaluate(x_test, y_test, batch_size=1024)
# print(f"loss : {loss} \n acc : {acc}")
model.summary()
print(f'loss_acc : {loss_acc}')
model.save(filepath='.\model\keras90_model\model_test01.h5')
# print(hist)
# print(hist.history.keys())
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
print(f'acc : {acc}')
print(f'val_acc : {val_acc}')
print(f'loss_acc : {loss_acc}')
# set the size of the graph
plt.figure(figsize=(10, 10))
# divide graphs to see clearly, x == epoch
# plot graph (2, 1, 1)
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() 
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'])
plt.legend(loc='upper right')
# plot graph (2, 1, 2)
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() 
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()
'''