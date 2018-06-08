import numpy as np
import tensorflow as tf


TRAINING_SET_PATH = 'cifar10_train.tfrecord'
TEST_SET_PATH = 'cifar10_test.tfrecord'


def _write_entries(images, labels, tfrecord_writer):

    # init computation graph
    tf.reset_default_graph()
    image_ph = tf.placeholder(dtype=tf.uint8)
    encode = tf.image.encode_png(image_ph, name='encoding')
   
    with tf.Session() as sess:
        for index in range(images.shape[0]):
            image = images[index]
            label = labels[index]

            label = tf.train.Feature(
                int64_list=tf.train.Int64List(value=label))
            encoding = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[sess.run(encode, feed_dict={image_ph: image})]))

            # create dataset entry
            entry = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': label,
                        'encoding': encoding
                    }
                )
            )
            tfrecord_writer.write(entry.SerializeToString())


def to_tfrecord(X_train, y_train, X_test, y_test):

    with tf.python_io.TFRecordWriter(TRAINING_SET_PATH) as writer:
        _write_entries(X_train, y_train, writer)
    
    with tf.python_io.TFRecordWriter(TRAINING_SET_PATH) as writer:
        _write_entries(X_test, y_test, writer)


if __name__ == '__main__':
    (X_train_, y_train_), (X_test_, y_test_) = tf.keras.datasets.cifar10.load_data()
    to_tfrecord(X_train_, y_train_, X_test_, y_test_)
