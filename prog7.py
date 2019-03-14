from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import time

import tensorflow as tf
from keras.utils import to_categorical


# Helper libraries
import matplotlib.pyplot as plt
from tensorflow.python.estimator import keras


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Load the train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# reshape tags
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

# Normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# To Categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Make model
model = tf.keras.Sequential()

# This  is the model suggested in HW, but accuray is low

# model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same', activation='relu', input_shape=(32, 32, 3)))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#
# model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#
# model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
# model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(10, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))


# Instead i used a model suggested from github

x = keras.layers.Input(shape=(32, 32, 3))
y = x
y = keras.layers.Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = keras.layers.Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)

y = keras.layers.Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = keras.layers.Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)

y = keras.layers.Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = keras.layers.Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')(y)

y = keras.layers.Flatten()(y)
y = keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal')(y)
y = keras.layers.Dropout(0.5)(y)
y = keras.layers.Dense(units=10, activation='softmax', kernel_initializer='he_normal')(y)

# Take a look at the model summary
model.summary()

# Compile the model, using optimizer adadelta
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model, save the model, print time used
start = time.time()
time_callback = TimeHistory()
myModel = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test), shuffle=True,
                    callbacks=[time_callback])
times = time_callback.times
print(times)

# plot training accuracy and loss graph


def plot_acc_loss(model, nb_epoch):
    acc, loss, val_acc, val_loss = model.history['acc'], model.history['loss'], model.history['val_acc'], model.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(nb_epoch), acc, label='Train')
    plt.plot(range(nb_epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(nb_epoch), loss, label='Train')
    plt.plot(range(nb_epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.savefig('plot.png')
    plt.show()


plot_acc_loss(myModel, 10)



