import argparse
import os

import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _make_tfrecord(record_name, root_path, ex_list):
    tf_file = tf.io.TFRecordWriter(record_name)
    for (img_path, label) in ex_list:
        with open(os.path.join(root_path, img_path), 'rb') as jpeg_file:
            jpeg_bytes = jpeg_file.read()
        if jpeg_bytes is None:
            print('{} is skipped because it cannot read the file.'.format(img_path))
            continue
        feature = {
            'image': _bytes_feature(jpeg_bytes),
            'label': _int64_feature(label)
        }
        exam = tf.train.Example(features=tf.train.Features(feature=feature))
        tf_file.write(exam.SerializeToString())
    tf_file.close()


def _parse_list_file(file_path):
    with open(file_path, 'r') as f:
        contents = f.readlines()
    contents = contents[2:] # discard two line
    train_list = []
    test_list = []
    train_label_db = {}
    test_label_db = {}
    train_label_counter = 0
    test_label_counter = 0
    for line in contents:
        sp = [item for item in line.splitlines()[0].split(' ') if item is not '']
        path = sp[0]
        is_train = sp[2]
        if is_train == 'train':
            if sp[1] in train_label_db:
                img_id = train_label_db[sp[1]]
            else:
                train_label_db[sp[1]] = train_label_counter
                img_id = train_label_counter
                train_label_counter += 1
            train_list.append((path, img_id))
        else:
            if sp[1] in test_label_db:
                img_id = test_label_db[sp[1]]
            else:
                test_label_db[sp[1]] = test_label_counter
                img_id = test_label_counter
                test_label_counter += 1
            test_list.append((path, img_id))
    return train_list, train_label_counter, test_list, test_label_counter


def make_inshop_tfrecord(root_path):
    list_file = os.path.join(root_path, 'eval/list_eval_partition.txt')
    train_list, train_label, test_list, test_label = _parse_list_file(list_file)

    _make_tfrecord('inshop_train.tfrecord', root_path, train_list)
    _make_tfrecord('inshop_test.tfrecord', root_path, test_list)

    print('generating tfrecord is finished.')
    print('# of images in train file: {}'.format(len(train_list)))
    print('# of labels in train file: {}'.format(train_label))
    print('# of images in test file: {}'.format(len(test_list)))
    print('# of labels in test file: {}'.format(test_label))


def load_tfrecord(train_file='inshop_train.tfrecord', test_file='inshop_test.tfrecord'):
    train_ds = tf.data.TFRecordDataset(train_file)
    test_ds = tf.data.TFRecordDataset(test_file)

    def _read_tfrecord(serialized):
        description = {
            'image': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64),
        }
        example = tf.io.parse_single_example(serialized, description)
        example['image'] = tf.io.decode_jpeg(example['image'], channels=3)
        return example

    train_ds = train_ds.map(_read_tfrecord)
    test_ds = test_ds.map(_read_tfrecord)
    return train_ds, test_ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
        help='absolute path of images in json_file')
    args = parser.parse_args()

    # all arguments must not be empty.
    if None in vars(args).values():
        parser.print_help()
    else:
        make_inshop_tfrecord(args.root_path)
