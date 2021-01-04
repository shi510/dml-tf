import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds


TF_AUTOTUNE = tf.data.AUTOTUNE


def random_flip(x: tf.Tensor):
    return tf.image.random_flip_left_right(x)


def random_color(x: tf.Tensor):
    x = tf.image.random_hue(x, 0.1)
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    return x


def cutout(x : tf.Tensor):
    const_rnd = tf.random.uniform([], 0., 1., dtype=tf.float32)
    size = tf.random.uniform([], 0, 4, dtype=tf.int32)
    size = size * 2
    return tfa.image.random_cutout(x, (size, size), const_rnd)


def attach_augmentation(ds):
    augmentations = [random_flip, random_color, cutout]
    for f in augmentations:
        choice = tf.random.uniform([], 0.0, 1.0)
        ds = ds.map(lambda x, label: (tf.cond(choice > 0.5, lambda: f(x), lambda: x), label),
            num_parallel_calls=TF_AUTOTUNE)
    ds = ds.map(lambda x, label: (tf.clip_by_value(x, 0., 1.), label), num_parallel_calls=TF_AUTOTUNE)
    return ds


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


def make_cub(batch_size, input_shape):
    train_ds, test_ds = tfds.load('caltech_birds2010', split=['train', 'test'])

    def _init(data):
        x = tf.image.resize(data['image'], input_shape)
        return (x, data['label'])


    def _transform_inputs(x, y):
        x = tf.cast(x, dtype=tf.float32) / 255.
        y = tf.cast(y, dtype=tf.int32)
        y = tf.reshape(y, (-1,))
        return (x, y)


    train_ds = train_ds.map(lambda data : _init(data), num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(lambda img, label : _transform_inputs(img, label), num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.prefetch(TF_AUTOTUNE)

    test_ds = test_ds.map(lambda data : _init(data), num_parallel_calls=TF_AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.map(lambda img, label : _transform_inputs(img, label), num_parallel_calls=TF_AUTOTUNE)
    test_ds = test_ds.prefetch(TF_AUTOTUNE)
    return train_ds, test_ds, 200


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


def make_cifar(batch_size, input_shape, classes):
    cifar_dataset = None
    if classes == 10:
        cifar_dataset = tf.keras.datasets.cifar10
    elif classes == 100:
        cifar_dataset = tf.keras.datasets.cifar100

    train_ds, test_ds = cifar_dataset.load_data()
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
    train_ds = attach_augmentation(train_ds)
    train_ds = train_ds.prefetch(TF_AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.map(lambda img, label : _transform_inputs(img, label), num_parallel_calls=TF_AUTOTUNE)
    test_ds = test_ds.prefetch(TF_AUTOTUNE)
    return train_ds, test_ds, classes


def make_tfdataset(dataset, batch_size, input_shape):
    if dataset == 'cars196':
        return make_cars196(batch_size, input_shape)
    elif dataset == 'mnist':
        return make_mnist(batch_size, input_shape)
    elif dataset == 'cifar10':
        return make_cifar(batch_size, input_shape, 10)
    elif dataset == 'cifar100':
        return make_cifar(batch_size, input_shape, 100)
    elif dataset == 'cub':
        return make_cub(batch_size, input_shape)
    else:
        print('dataset {} not supports.'.format(dataset))