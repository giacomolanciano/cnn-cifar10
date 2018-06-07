import pickle
from datetime import timedelta

import os
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from to_tfrecord import TRAINING_SET_PATH, TEST_SET_PATH


CIFAR10_CLASSES = 10
CIFAR10_TRAIN_SIZE = 50000
CIFAR10_TEST_SIZE = 10000
EPOCHS = 1000
BATCH_SIZE = 150
MODEL_CHECKPOINT_SAMPLING = 100
ACCURACY_SAMPLING = 10
CHECKPOINT_FILENAME = 'cnn.ckpt'
KEEP_PROB = 0.5


def _parse_dataset_features(entry):
    features = {
        'label': tf.FixedLenFeature((), dtype=tf.int64, default_value=0),
        'encoding': tf.FixedLenFeature((), dtype=tf.string, default_value='')
    }
    features_dict = tf.parse_single_example(entry, features)
    label = tf.one_hot(features_dict['label'], depth=CIFAR10_CLASSES, dtype=tf.float32)
    image = tf.image.convert_image_dtype(tf.image.decode_png(features_dict['encoding']), dtype=tf.float32)
    return image, label


def load_dataset(dataset_filename, shuffle_buffer=None, batch_size=None, repeat=None):
    dataset = tf.data.TFRecordDataset(dataset_filename)
    dataset = dataset.map(_parse_dataset_features)
    if shuffle_buffer:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    if batch_size:
        dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat(repeat)
    return dataset


def plot_curve(curve, fig_name, fig_ext='.png'):
    fig = fig_name + fig_ext
    fig_zoom = fig_name + '_zoom' + fig_ext

    measures_num = len(curve)
    plt.figure()
    plt.plot(range(1, measures_num + 1), curve)
    plt.axis([1, measures_num, 0, 1])
    plt.savefig(fig, bbox_inches='tight')

    plt.figure()
    plt.plot(range(1, measures_num + 1), curve)
    plt.savefig(fig_zoom, bbox_inches='tight')


def make_results_dir():
    dirpath = 'data_' + str(int(time.time()))
    os.makedirs(dirpath)
    return dirpath


def dump_results(results, results_filename):
    pickle.dump(results, open(results_filename, 'wb'))


def main(argv=None):
    tf.reset_default_graph()

    # variables
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input')
    y = tf.placeholder(tf.float32, [None, 10], name='labels')
    keep_prob = tf.placeholder(tf.float32)

    ####################################################################################################################

    # init computation graph
    conv1 = tf.layers.conv2d(
        X, filters=128, kernel_size=(5, 5), strides=1, activation=tf.nn.relu, padding='same', name='conv1')

    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=2, padding='same', name='maxpool1')

    conv2 = tf.layers.conv2d(
        maxpool1, filters=128, kernel_size=(5, 5), strides=1, activation=tf.nn.relu, padding='same', name='conv2')

    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2, padding='same', name='maxpool2')

    conv3 = tf.layers.conv2d(
        maxpool2, filters=128, kernel_size=(5, 5), strides=1, activation=tf.nn.relu, padding='same', name='conv3')

    maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=2, padding='same', name='maxpool2')

    conv4 = tf.layers.conv2d(
        maxpool3, filters=128, kernel_size=(5, 5), strides=1, activation=tf.nn.relu, padding='same', name='conv4')

    maxpool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=2, padding='same', name='maxpool2')

    conv5 = tf.layers.conv2d(
        maxpool4, filters=128, kernel_size=(5, 5), strides=1, activation=tf.nn.relu, padding='same', name='conv5')

    maxpool5 = tf.layers.max_pooling2d(conv5, pool_size=(2, 2), strides=2, padding='same', name='maxpool5')

    dense1 = tf.layers.dense(
        tf.reshape(maxpool5, [-1, 128]), units=4096, activation=tf.nn.relu, name='dense1')
    dense1_dropout = tf.nn.dropout(dense1, keep_prob=keep_prob, name='dense1_dropout')

    dense2 = tf.layers.dense(dense1_dropout, units=4096, activation=tf.nn.relu, name='dense2')
    dense2_dropout = tf.nn.dropout(dense2, keep_prob=keep_prob, name='dense2_dropout')

    output = tf.layers.dense(dense2_dropout, units=CIFAR10_CLASSES, name='output')

    # functions
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss, global_step=global_step)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32), name='accuracy')

    ####################################################################################################################

    RESULTS_DIR = make_results_dir()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load training dataset
        train_dataset = load_dataset(
            TRAINING_SET_PATH, shuffle_buffer=CIFAR10_TRAIN_SIZE // 2, batch_size=BATCH_SIZE, repeat=-1)
        iterator = train_dataset.make_one_shot_iterator()
        next_elem = iterator.get_next()

        # training sessions
        model_saver = tf.train.Saver()
        training_accuracy_curve = []
        start_time = time.time()
        for epoch in range(EPOCHS):
            print('##### Epoch {} #####'.format(epoch))

            # get batch from dataset
            try:
                X_batch, y_batch = sess.run(next_elem)

                # train network
                sess.run(train_step, feed_dict={X: X_batch, y: y_batch, keep_prob: KEEP_PROB})

                # compute accuracy
                if epoch % ACCURACY_SAMPLING == 0 or epoch == EPOCHS - 1:
                    epoch_accuracy = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch, keep_prob: 1.0})
                    training_accuracy_curve.append(epoch_accuracy)
                    print('accuracy: {}'.format(epoch_accuracy))

                # dump current state
                if epoch % MODEL_CHECKPOINT_SAMPLING == 0 or epoch == EPOCHS - 1:
                    checkpoint_path = os.path.join(RESULTS_DIR, CHECKPOINT_FILENAME)
                    model_saver.save(sess, checkpoint_path, global_step=epoch)

            except tf.errors.OutOfRangeError:
                print('Training set has been consumed.')

        # collect training results
        elapsed_time = (time.time() - start_time)
        training_seconds = timedelta(seconds=elapsed_time)
        print('training time:', training_seconds, 'seconds')
        print('training accuracy curve:', training_accuracy_curve)
        plot_curve(training_accuracy_curve, fig_name=os.path.join(RESULTS_DIR, 'train_accuracy'))

    ####################################################################################################################

        print('\n\n##### Validation #####')
        # load test dataset
        test_dataset = load_dataset(TEST_SET_PATH, batch_size=100)
        iterator = test_dataset.make_one_shot_iterator()
        next_elem = iterator.get_next()

        test_accuracy_collection = []
        try:
            while True:
                X_test, y_test = sess.run(next_elem)
                test_accuracy_collection.append(sess.run(accuracy, feed_dict={X: X_test, y: y_test, keep_prob: 1.0}))
        except tf.errors.OutOfRangeError:
            print('Test set has been consumed.')

        test_accuracy = sess.run(tf.reduce_mean(tf.convert_to_tensor(test_accuracy_collection)))
        print('test accuracy:', test_accuracy)

        # store results
        results = dict()
        results['training_seconds'] = training_seconds
        results['training_accuracy_curve'] = training_accuracy_curve
        results['test_accuracy'] = test_accuracy
        dump_results(results, os.path.join(RESULTS_DIR, 'results.pickle'))


if __name__ == '__main__':
    tf.app.run()
