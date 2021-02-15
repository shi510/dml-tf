import dataset.inshop

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


def random_cutout(x : tf.Tensor):
    const_rnd = tf.random.uniform([], 0., 1., dtype=tf.float32)
    size = tf.random.uniform([], 0, 30, dtype=tf.int32)
    size = size * 2
    return tfa.image.random_cutout(x, (size, size), const_rnd)


def attach_augmentation(ds):
    augmentations = [random_flip, random_color, random_cutout]
    for f in augmentations:
        choice = tf.random.uniform([], 0.0, 1.0)
        ds = ds.map(lambda x, label: (tf.cond(choice > 0.5, lambda: f(x), lambda: x), label),
            num_parallel_calls=TF_AUTOTUNE)
    ds = ds.map(lambda x, label: (tf.clip_by_value(x, 0., 1.), label), num_parallel_calls=TF_AUTOTUNE)
    return ds


def make_dataset(train_ds, test_ds, batch_size, input_shape, image_key='image', label_key='label'):


    def _init(data, crop_shape, training=False):
        x = data[image_key]
        x = tf.image.resize(x, (256, 256))
        if training:
            x = tf.image.random_crop(x, crop_shape)
            x = tf.image.random_flip_left_right(x)
        else:
            x = tf.image.central_crop(x, input_shape[0] / 256.)
        x = tf.cast(x, dtype=tf.float32) / 255.
        y = tf.cast(data[label_key], dtype=tf.int32)
        return x, y


    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.map(
        lambda data : _init(data, input_shape, True),
        num_parallel_calls=TF_AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(TF_AUTOTUNE)

    test_ds = test_ds.map(
        lambda data : _init(data, input_shape, False),
        num_parallel_calls=TF_AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(TF_AUTOTUNE)

    return train_ds, test_ds


def make_tfdataset(name, batch_size, input_shape):
    dataset_list = {
        'cars196': ('cars196', 'image', 'label', 196, 196, ['train', 'test']),
        'cub': ('caltech_birds2011', 'image', 'label', 200, 200, ['train', 'test']),
        'sop': ('StanfordOnlineProducts', 'image', 'class_id', 11318, 11316, ['train', 'test']),
        'inshop': (None, 'image', 'label', 3997, 3985, None)
    }
    if name not in dataset_list:
        raise 'dataset {} not supports.'.format(name)

    tfds_name, img_key, lb_key, train_classes, test_classes, split = dataset_list[name]
    if name == 'inshop':
        # inshop data should be downloaded manually.
        train_ds, test_ds = dataset.inshop.load_tfrecord()
    else:
        train_ds, test_ds = tfds.load(tfds_name, split=split)
    train_ds, test_ds = make_dataset(train_ds, test_ds, batch_size,
        input_shape, img_key, lb_key)
    return train_ds, train_classes, test_ds, test_classes
