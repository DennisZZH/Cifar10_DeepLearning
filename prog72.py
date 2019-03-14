from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow import keras

# Load the train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# reshape tags
y_test = y_test.reshape(y_test.shape[0])

# Normalize
x_test = x_test.astype('float32')
x_test /= 255

# To Categorical
y_test = to_categorical(y_test, 10)

model = keras.models.load_model('CIFAR10_model_no_data_augmentation.h5_best')


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Error rate = %.2f %% " % ((1-test_acc) * 100))

