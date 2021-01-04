import tensorflow as tf
import tensorflow_datasets as tfds


TF_AUTOTUNE = tf.data.AUTOTUNE


def make_cars196(batch_size, input_shape):
    train_ds, test_ds = tfds.load('cars196', split=['train', 'test'])

    def _init(data):
        x = tf.image.resize(data['image'], input_shape)
        return (x, data['label'])

    def _transform_inputs(x, y):
        x = tf.cast(x, dtype=tf.float32) / 255.
        y = tf.cast(y, dtype=tf.int32)
        y = tf.reshape(y, (-1,))
        return (x, y)

    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.map(lambda data : _init(data), num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(lambda img, label : _transform_inputs(img, label), num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.prefetch(TF_AUTOTUNE)

    test_ds = test_ds.map(lambda data : _init(data), num_parallel_calls=TF_AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.map(lambda img, label : _transform_inputs(img, label), num_parallel_calls=TF_AUTOTUNE)
    test_ds = test_ds.prefetch(TF_AUTOTUNE)
    return train_ds, test_ds, 196


def make_mnist(batch_size, input_shape):
    train_ds, test_ds = tf.keras.datasets.mnist.load_data()
    train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)

    def _transform_inputs(x, y):
        x = tf.cast(x, dtype=tf.float32) / 255.
        y = tf.cast(y, dtype=tf.int32)
        y = tf.reshape(y, (-1,))
        return (x, y)

    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(lambda img, label : _transform_inputs(img, label), num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.prefetch(TF_AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.map(lambda img, label : _transform_inputs(img, label), num_parallel_calls=TF_AUTOTUNE)
    test_ds = test_ds.prefetch(TF_AUTOTUNE)
    return train_ds, test_ds, 10


def make_cifar10(batch_size, input_shape):
    train_ds, test_ds = tf.keras.datasets.cifar10.load_data()
    train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)

    def _transform_inputs(x, y):
        x = tf.cast(x, dtype=tf.float32) / 255.
        y = tf.cast(y, dtype=tf.int32)
        y = tf.reshape(y, (-1,))
        return (x, y)

    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(lambda img, label : _transform_inputs(img, label), num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.prefetch(TF_AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.map(lambda img, label : _transform_inputs(img, label), num_parallel_calls=TF_AUTOTUNE)
    test_ds = test_ds.prefetch(TF_AUTOTUNE)
    return train_ds, test_ds, 10


def make_tfdataset(dataset, batch_size, input_shape):
    if dataset == 'cars196':
        return make_cars196(batch_size, input_shape)
    elif dataset == 'mnist':
        return make_mnist(batch_size, input_shape)
    elif dataset == 'cifar10':
        return make_cifar10(batch_size, input_shape)
    else:
        print('dataset {} not supports.'.format(dataset))