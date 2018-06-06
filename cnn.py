import tensorflow as tf
from to_tfrecord import TRAINING_SET_PATH, TEST_SET_PATH


EPOCHS = 1000
BATCH_SIZE = 128
CIFAR10_TEST_SIZE = 10000
KEEP_PROB = 0.5
CHECKPOINT_PATH = 'cnn.ckpt'


def _parse_dataset_features(entry):
    features = {
        'label': tf.FixedLenFeature((), dtype=tf.int64, default_value=0),
        'encoding': tf.FixedLenFeature((), dtype=tf.string, default_value='')
    }
    features_dict = tf.parse_single_example(entry, features)
    label = features_dict['label']
    image = tf.image.convert_image_dtype(tf.image.decode_png(features_dict['encoding']), dtype=tf.float32)
    return image, label


def load_dataset(dataset_filename, batch_size=None):
    dataset = tf.data.TFRecordDataset(dataset_filename)
    dataset = dataset.map(_parse_dataset_features)
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


def main(argv=None):
    # variables
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

    # placeholders
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input')
    y = tf.placeholder(tf.float32, [None, 10], name='labels')
    keep_prob = tf.placeholder(tf.float32)

    ####################################################################################################################

    # init computation graph
    conv1 = tf.layers.conv2d(
        X, filters=64, kernel_size=(5, 5), strides=1, activation=tf.nn.relu, padding='same', name='conv1')

    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=1, padding='same', name='maxpool1')

    conv2 = tf.layers.conv2d(
        maxpool1, filters=128, kernel_size=(5, 5), strides=1, activation=tf.nn.relu, padding='same', name='conv2')

    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2, padding='same', name='maxpool2')

    conv3 = tf.layers.conv2d(
        maxpool2, filters=192, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding='same', name='conv3')

    conv4 = tf.layers.conv2d(
        conv3, filters=192, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding='same', name='conv4')

    conv5 = tf.layers.conv2d(
        conv4, filters=128, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding='same', name='conv5')

    maxpool3 = tf.layers.max_pooling2d(conv5, pool_size=(2, 2), strides=1, padding='same', name='maxpool3')

    dense1 = tf.layers.dense(
        tf.reshape(maxpool3, [-1, 16 * 16 * 128]), units=4096, activation=tf.nn.relu, name='dense1')
    dense1 = tf.nn.dropout(dense1, keep_prob=keep_prob)

    dense2 = tf.layers.dense(dense1, units=4096, activation=tf.nn.relu, name='dense2')
    dense2 = tf.nn.dropout(dense2, keep_prob=keep_prob)

    output = tf.layers.dense(dense2, units=10, activation=tf.nn.relu, name='output')

    ####################################################################################################################

    # functions
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32), name='accuracy')

    # training sessions
    model_saver = tf.train.Saver()
    accuracy_curve = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # load TFRecordDataset
        train_dataset = load_dataset(TRAINING_SET_PATH, batch_size=BATCH_SIZE)
        iterator = train_dataset.make_one_shot_iterator()

        for epoch in range(EPOCHS):
            # get batch from dataset
            X_batch, y_batch = iterator.get_next()

            sess.run(train_step, feed_dict={X: X_batch, y: y_batch, keep_prob: KEEP_PROB})

            if epoch % 100 == 0:
                epoch_accuracy = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch, keep_prob: 1.0})
                accuracy_curve.append(epoch_accuracy)
                print('accuracy: {}'.format(epoch_accuracy))

                # dump current state
                model_saver.save(sess, CHECKPOINT_PATH, global_step=epoch)

        # get test split from dataset
        test_dataset = load_dataset(TEST_SET_PATH, batch_size=CIFAR10_TEST_SIZE)
        iterator = test_dataset.make_one_shot_iterator()
        X_test, y_test = iterator.get_next()

        # validation
        test_accuracy = sess.run(accuracy, feed_dict={X: X_test, y: y_test, keep_prob: 1.0})
        print('test accuracy: {}'.format(test_accuracy))


if __name__ == '__main__':
    tf.app.run()
