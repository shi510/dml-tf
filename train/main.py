import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import train.input_pipeline as input_pipeline
import train.config
import net_arch.models
from train.callbacks import ConfusionMatrixCallback, LogCallback
from train.callbacks import RecallCallback
from train.callbacks import NMICallback
from train.custom_model import ProxyNCAModel
from train.custom_model import ProxyAnchorModel
import train.loss.utils

import tensorflow as tf
import tensorflow_addons as tfa


def build_dataset(config):
    train_ds, train_classes, test_ds, test_classes = input_pipeline.make_tfdataset(
        config['dataset'],
        config['batch_size'],
        config['shape'])
    return train_ds, train_classes, test_ds, test_classes


def build_backbone_model(config):
    return net_arch.models.get_model(config['model'], config['shape'])


def build_embedding_model(config, n_classes):
    y = x = tf.keras.Input(config['shape'])
    y = build_backbone_model(config)(y)

    def _embedding_layer(feature):
        y = x = tf.keras.Input(feature.shape[1:])
        y = tf.keras.layers.Dropout(rate=0.3)(y)
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        y = tf.keras.layers.Dense(config['embedding_dim'])(y)
        y = tf.keras.layers.BatchNormalization()(y)
        return tf.keras.Model(x, y, name='embeddings')(feature)


    y = _embedding_layer(y)
    loss_param = copy.deepcopy(config['loss_param'][config['loss']])
    loss_param['n_embeddings'] = config['embedding_dim']
    loss_param['n_classes'] = n_classes
    if config['loss'] == 'ProxyNCA':
        return ProxyNCAModel(inputs=x, outputs=y, **loss_param)
    elif config['loss'] == 'ProxyAnchor':
        return ProxyAnchorModel(inputs=x, outputs=y, **loss_param)
    else:
        raise 'Not supported loss'


def build_callbacks(config, model, test_ds, monitor, mode):
    log_dir = os.path.join('logs', config['model_name'])
    callback_list = []
    top_k = config['eval']['recall']
    metric = config['eval']['metric']
    test_recall = RecallCallback(test_ds, top_k, metric, log_dir)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=0.1, verbose=1,
        patience=3, min_lr=1e-4, mode=mode)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir,
        save_weights_only=False,
        monitor=monitor,
        mode=mode,
        save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=5, mode=mode)
    tensorboard_log = LogCallback(log_dir)
    cm = ConfusionMatrixCallback(model.proxy_loss.proxies, log_dir)

    callback_list.append(test_recall)
    callback_list.append(tensorboard_log)
    if not config['lr_decay']:
        callback_list.append(reduce_lr)
    callback_list.append(early_stop)
    callback_list.append(checkpoint)
    callback_list.append(cm)
    return callback_list, early_stop


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
    return opt_list[config['optimizer']](config['lr'])


def remove_subclassing_keras_model(model):
    y = x = model.layers[0].input
    for l in model.layers[1:]:
        y = l(y)
    return tf.keras.Model(x, y, name=model.name)


def start_training(config):
    train_ds, train_classes, test_ds, test_classes = build_dataset(config)
    net = build_embedding_model(config, train_classes)
    opt = build_optimizer(config)
    callbacks, early_stop = build_callbacks(config, net, test_ds, 'recall@1', 'max')
    net.summary()
    net.compile(optimizer=opt)
    net.fit(train_ds, epochs=config['epoch'], verbose=1,
        workers=input_pipeline.TF_AUTOTUNE,
        callbacks=callbacks)
    if early_stop.best_weights is not None:
        net.set_weights(early_stop.best_weights)
    if config['eval']['NMI']:
        test_nmi = NMICallback(test_ds, test_classes,
            os.path.join('logs', config['model_name']))
        test_nmi.set_model(net)
        test_nmi.on_train_end()
    net = remove_subclassing_keras_model(net)
    net.save('{}.h5'.format(config['model_name']), include_optimizer=False)

if __name__ == '__main__':
    config = train.config.config
    start_training(config)
