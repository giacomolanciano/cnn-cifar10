import numpy as np
import tensorflow as tf


TRAINING_SET_PATH = 'cifar10_train.tfrecord'
TEST_SET_PATH = 'cifar10_test.tfrecord'


def _create_entry(sample, sample_label):

    # init computation graph
    tf.reset_default_graph()
    image = tf.placeholder(dtype=tf.uint16)
    encode = tf.image.encode_png(image, name='encoding')

    with tf.Session() as sess:
        height = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[sample.shape[0]]))
        width = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[sample.shape[1]]))
        channels = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[sample.shape[2]]))
        label = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[sample_label]))
        encoding = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[sess.run(encode, feed_dict={image: sample})]))

    # create dataset entry
    entry = tf.train.Example(
        features=tf.train.Features(
            feature={
                'height': height,
                'width': width,
                'channels': channels,
                'label': label,
                'encoding': encoding
            }
        )
    )
    return entry


def to_tfrecord(X_train, y_train, X_test, y_test):

    with tf.python_io.TFRecordWriter(TRAINING_SET_PATH) as writer:
        for index in range(X_train.shape[0]):
            sample = np.squeeze(X_train[index])
            label = np.squeeze(y_train[index])
            writer.write(_create_entry(sample, label).SerializeToString())

    with tf.python_io.TFRecordWriter(TEST_SET_PATH) as writer:
        for index in range(X_test.shape[0]):
            sample = np.squeeze(X_test[index])
            label = np.squeeze(y_test[index])
            writer.write(_create_entry(sample, label).SerializeToString())


if __name__ == '__main__':
    (X_train_, y_train_), (X_test_, y_test_) = tf.keras.datasets.cifar10.load_data()
    to_tfrecord(X_train_, y_train_, X_test_, y_test_)
