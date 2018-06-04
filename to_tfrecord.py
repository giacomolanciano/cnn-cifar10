import os
import tensorflow as tf


TRAINING_SET_PATH = 'cifar10_train.tfrecord'
TEST_SET_PATH = 'cifar10_test.tfrecord'


def to_tfrecord(X_train, y_train, X_test, y_test):
    pass


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
