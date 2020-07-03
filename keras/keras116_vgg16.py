from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D
from keras.layers import Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

vgg16 = VGG16()
# take_model = ResNet101()
# take_model = ResNet101V2()
# take_model = ResNet152()
# take_model = ResNet152V2()
# take_model = ResNet50()
# take_model = ResNet50V2()
# take_model = InceptionV3()
# take_model = InceptionResNetV2()

# VGG-16 모델은 Image Net Challenge에서 Top-5 테스트 정확도를 92.7% 달성

vgg16 = VGG16()
vgg19 = VGG19()

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# model = Sequential()
# model.add(vgg19)
# # model.add(Flatten())
# model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()

