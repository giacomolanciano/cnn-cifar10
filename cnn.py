import numpy as np
import tensorflow as tf


class CNNModel:

    def prediction(self, X):
        # conv1 + relu
        with tf.variable_scope('conv1') as scope:
            conv1 = tf.nn.conv2d(X, padding='SAME')

        # maxpooling1

        # conv2 + relu

        # maxpooling2

        # conv3 + relu

        # conv4 + relu

        # conv5 + relu

        # maxpooling3

        # dense1

        # dense2

        # dense3

        # softmax
        pass


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
