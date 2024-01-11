# %%

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Input, Lambda, Dropout

class SimpleCNN:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Input(shape=(shape[0], shape[1], shape[2])))
        #model.add(Lambda(lambda x: expand_dims(x, axis=-1)))
        model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Activation("relu"))
        model.add(Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Activation("relu"))
        model.add(Conv2D(filters=512, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=512, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

model = SimpleCNN.build((28, 28, 3), 10)
model.summary()

class SimpleMLP:
    @staticmethod
    def build(shape): #, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        # model.add(Dense(classes))
        # model.add(Activation("softmax"))
        model.add(Dense(1))
        return model

class CNN_cifar10:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        # model.add(Input(shape=(shape[0], shape[1], shape[2])))
        model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation="relu"))
        model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))
        model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation="relu"))
        model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1024,activation="relu"))
        model.add(Dense(1024,activation="relu"))
        model.add(Dense(units = 10, activation = "softmax"))
        return model
        