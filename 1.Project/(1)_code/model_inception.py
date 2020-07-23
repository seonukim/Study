import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import InceptionV3, MobileNet, Xception
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

start = time.time()

# load data
x = np.load('C:/Users/bitcamp/Downloads/data/face_image_total.npy')
y = np.load('C:/Users/bitcamp/Downloads/data/face_label_total.npy')

print(x.shape) # (1414, 112, 112, 3)
print(y.shape) # (1414, 14)
print('data_load 걸린 시간 :', time.time() - start)
print('======== data load ========')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

# model
takemodel = Xception(include_top = False, input_shape = (128, 128, 3))

model = Sequential()
model.add(takemodel)
model.add(Flatten())
model.add(Dense(120, activation = 'softmax'))

model.summary()

cp = ModelCheckpoint('D:/Study/1.Project/(3)_Saved_model/Xception_batch128.hdf5',
                     monitor = 'val_loss', save_best_only = True, save_weights_only = False)
es = EarlyStopping(monitor= 'val_loss', patience = 50, verbose =1)

#3. compile, fit
model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])                             
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 128, verbose = 1, 
                 validation_split = 0.2, shuffle = True, callbacks = [es, cp])


#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss_acc: ' ,loss_acc)

end = time.time()
print('총 걸린 시간 :', end - start)

import matplotlib.pyplot as plt
plt.figure(figsize = (10, 6))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '^', c = 'magenta', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '^', c = 'cyan', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = '^', c = 'magenta', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '^', c = 'cyan', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()





