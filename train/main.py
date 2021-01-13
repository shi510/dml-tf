import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import train.input_pipeline as input_pipeline
import train.config
import net_arch.models
from train.callbacks import LogCallback, RecallCallback, ReduceLROnPlateau
from train.loss.proxynca import ProxyNCALoss
from train.loss.triplet import original_triplet_loss as triplet_loss
from train.utils import CustomModel

import tensorflow as tf
import tensorflow_addons as tfa


def build_dataset(config):
    train_ds, test_ds, classes = input_pipeline.make_tfdataset(
        config['dataset'],
        config['batch_size'],
        config['shape'])
    return train_ds, test_ds, classes


def build_backbone_model(config):
    return net_arch.models.get_model(config['model'], config['shape'])


def build_embedding_model(config):
    y = x = tf.keras.Input(config['shape'])
    y = build_backbone_model(config)(y)

    def _embedding_layer(feature):
        y = x = tf.keras.Input(feature.shape[1:])
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        y = tf.keras.layers.Dense(config['embedding_dim'])(y)
        return tf.keras.Model(x, y, name='embeddings')(feature)


    y = _embedding_layer(y)

    return tf.keras.Model(x, y, name=config['model_name'])


def build_callbacks(config, test_ds, optimizer, monitor, mode):
    callback_list = []
    model_name = config['model_name']
    log_dir = os.path.join('logs', model_name)
    if test_ds is not None:
        top_k = config['eval']['recall']
        callback_list.append(RecallCallback(test_ds, top_k, log_dir))
    reduce_lr_net = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=0.1, verbose=1,
        patience=5, min_lr=1e-4, mode=mode)
    reduce_lr_proxy = ReduceLROnPlateau(
        optimizer=optimizer,
        monitor=monitor, factor=0.1, verbose=1,
        patience=5, min_lr=1e-4, mode=mode)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint'+os.path.sep+config['model_name'],
        save_weights_only=False,
        monitor=monitor,
        mode=mode,
        save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=10, mode=mode)
    tensorboard_log = LogCallback(log_dir)

    if not config['lr_decay']:
        callback_list.append(reduce_lr_net)
        callback_list.append(reduce_lr_proxy)
    # callback_list.append(checkpoint)
    callback_list.append(early_stop)
    callback_list.append(tensorboard_log)
    return callback_list


def build_optimizer(config):
    opt_list = {
        'Adam': 
            lambda lr:
            tf.keras.optimizers.Adam(learning_rate=lr),
        'AdamW': 
            lambda lr:
            tfa.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        'RMSprop':
            lambda lr:
                tf.keras.optimizers.RMSprop(learning_rate=lr),
        'SGD':
            lambda lr:
                tf.keras.optimizers.SGD(learning_rate=lr,
                    momentum=0.9, nesterov=True)
    }

    if config['optimizer'] not in opt_list:
        print(config['optimizer'], 'is not support.')
        print('please select one of below.')
        print(opt_list.keys())
        exit(1)
    loss_param = config['loss_param'][config['loss']]
    model_opt = opt_list[config['optimizer']](config['lr'])
    proxy_opt = opt_list[config['optimizer']](loss_param['lr'])
    return [model_opt, proxy_opt]


def build_loss(config, n_class):
    loss_fn = None
    config = copy.deepcopy(config)
    loss_param = config['loss_param'][config['loss']]
    n_embedding = config['embedding_dim']
    del(loss_param['lr'])
    if config['loss'] == 'ProxyNCA':
        loss_fn = ProxyNCALoss(n_embedding, n_class, **loss_param)
    else:
        raise 'The Loss -> {} is not supported.'.format(config['loss'])

    return loss_fn


def start_training(config):
    train_ds, test_ds, classes = build_dataset(config)
    net = build_embedding_model(config)
    loss_fn = build_loss(config, classes)
    opt_list = build_optimizer(config)
    callbacks = build_callbacks(config, test_ds, opt_list[1], 'recall@1', 'max')
    net.summary()
    cm = CustomModel(net, loss_fn, opt_list[0])
    cm.add_optimizer(opt_list[1], loss_fn.trainable_weights)
    cm.fit(train_ds, config['epoch'], callbacks)


if __name__ == '__main__':
    config = train.config.config
    start_training(config)
