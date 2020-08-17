import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import InceptionV3, MobileNet, Xception 
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import HDF5Matrix
import time
import efficientnet.tfkeras as efn 
import tensorflow as tf
import h5py
# tf.compat.v1.keras.applications.EfficientNetB2

start = time.time()

# load data
path = 'D:/data/hdf5/DATA.hdf5'
f = h5py.File(path)
x = f['256'][:]
y = f['label'][:]

print(x.shape) # (7481, 512, 512, 3)
print(y.shape) # (7481,  )
print('data_load 걸린 시간 :', time.time() - start)
print('======== data load ========')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state = 66)

# model
def cnn_model():
    takemodel = efn.EfficientNetB2(weights='imagenet',include_top = False, input_shape = (256, 256, 3))
    takemodel.trainable = True
    takemodel.summary()
    layer_dict = dict([(layer.name, layer) for layer in takemodel.layers])

    x = layer_dict['top_activation'].output
    print(x)
    
    x = Conv2D(filters=300, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='softmax')(x)

    model = Model(takemodel.input,x )


    # x = Dens(16,activation = "softmax")(takemodel)

    # model = Model(inputs = )

    # model.add(takemodel)
    # model.add(Flatten())
    # model.add(Dense(16, activation = 'softmax'))

    model.summary()

    return model

# Multi-GPU model
from tensorflow.keras.utils import multi_gpu_model
model = cnn_model()
model = multi_gpu_model(model, gpus = 2)

cp = ModelCheckpoint('D:/checkpoint/efficientnet_true1.hdf5', monitor = 'val_loss',
                    save_best_only = True, save_weights_only = False)
es = EarlyStopping(monitor= 'val_loss', patience = 25, verbose =1)

#3. compile, fit
model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])                             
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 64, verbose = 1,
                validation_split =0.3 ,shuffle = True,
                callbacks = [es, cp])


#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss_acc: ' ,loss_acc)

end = time.time()
print('총 걸린 시간 :', end-start)

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
plt.ylabel('acc')
plt.legend()

plt.show()