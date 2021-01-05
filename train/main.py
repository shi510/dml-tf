import os

import train.input_pipeline as input_pipeline
import train.config
from train.callbacks import LossTensorBoard
import net_arch.models
from train.loss.proxynca import ProxyNCALoss
from train.loss.triplet import original_triplet_loss as triplet_loss
import evalutate.nmi as nmi
import evalutate.linear as linear_eval

import tensorflow as tf
import numpy as np


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
        y = tf.keras.layers.Dropout(rate=0.5)(y)
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        y = tf.keras.layers.Dense(config['embedding_dim'])(y)
        return tf.keras.Model(x, y, name='embeddings')(feature)


    y = _embedding_layer(y)

    return tf.keras.Model(x, y, name=config['model_name'])


def build_callbacks(config):
    callback_list = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1,
        patience=1, min_lr=1e-4)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint'+os.path.sep+config['model_name'],
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    tensorboard_log = LossTensorBoard(
        100, os.path.join('logs', config['model_name']))

    if not config['lr_decay']:
        callback_list.append(reduce_lr)
    # callback_list.append(checkpoint)
    callback_list.append(early_stop)
    callback_list.append(tensorboard_log)
    return callback_list


def build_optimizer(config):
    # In tf-v2.3.0, Do not use tf.keras.optimizers.schedules with ReduceLR callback.
    if config['lr_decay']:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            config['lr'],
            decay_steps=config['lr_decay_steps'],
            decay_rate=config['lr_decay_rate'],
            staircase=True)
    else:
        lr = config['lr']

    opt_list = {
        'adam': 
            tf.keras.optimizers.Adam(learning_rate=lr),
        'sgd':
            tf.keras.optimizers.SGD(learning_rate=lr,
                momentum=0.9, nesterov=True)
    }
    if config['optimizer'] not in opt_list:
        print(config['optimizer'], 'is not support.')
        print('please select one of below.')
        print(opt_list.keys())
        exit(1)
    return opt_list[config['optimizer']]


def build_loss(config):
    loss_fn = None
    loss_param = config['loss_param']
    if config['loss'] == 'ProxyNCA':
        proxy_param = loss_param['ProxyNCA']
        embedding_scale = proxy_param['embedding_scale']
        proxy_scale = proxy_param['proxy_scale']
        loss_fn = ProxyNCALoss(config['embedding_dim'], classes,
            embedding_scale, proxy_scale)
    else:
        raise 'The Loss -> {} is not supported.'.format(config['loss'])
    
    return loss_fn


if __name__ == '__main__':
    config = train.config.config
    train_ds, test_ds, classes = build_dataset(config)
    net = build_embedding_model(config)
    loss_fn = build_loss(config)
    opt = build_optimizer(config)
    net.summary()

    # Iterate over epochs.
    for epoch in range(config['epoch']):
        print('Epoch %d' % epoch)
        # Iterate over the batches of the dataset.
        epoch_loss = np.zeros(1, dtype=np.float32)
        total_iter = 0
        for step, (x, y) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                probs = net(x)
                total_loss = loss_fn(y, probs)
            epoch_loss += total_loss.numpy()
            all_weights = net.trainable_weights + [loss_fn.proxies]
            grads = tape.gradient(total_loss, all_weights)
            opt.apply_gradients(zip(grads, all_weights))
            total_iter += 1
        nmi_score = nmi.evaluate(net, test_ds, config['embedding_dim'], classes)
        print('training loss={}'.format(epoch_loss / total_iter))
        print('test set NMI={:.3f}%'.format(nmi_score*100))
        # mean, var = tf.nn.moments(loss_fn.proxies, 0)
        # print('proxy mean=', mean.numpy())
        # print('proxy var=', var.numpy())
    linear_eval_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    linear_eval.evaluate(net, classes, train_ds, test_ds, linear_eval_opt)
