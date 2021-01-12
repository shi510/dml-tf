import os
import itertools

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


class CallbackList(object):

    def __init__(self, model, callbacks):
        self.model = model
        self.callbacks = callbacks
        for cb in self.callbacks:
            cb.set_model(self.model)


    def on_batch_begin(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)


    def on_batch_end(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)


    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)


    def on_epoch_end(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)


    def on_train_begin(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)


    def on_train_end(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_end(logs)


class CustomModel(object):
    
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.trainable_weights = [self.model.trainable_weights]
        self.criterion = loss
        self.optmizers = [optimizer]
        self.model.optimizer = optimizer


    def add_optimizer(self, optimizer, variables):
        self.optmizers.append(optimizer)
        if not isinstance(variables, list):
            raise 'variables should be a list type.'
        self.trainable_weights.append(variables)


    def fit(self, train_ds, epoch, callbacks):
        callback_list = CallbackList(self.model, callbacks)
        callback_list.on_train_begin()
        for n in range(epoch):
            print('Epoch %d' % n)
            callback_list.on_epoch_begin(n)
            pbar = tf.keras.utils.Progbar(len(train_ds))
            logs = {}
            # Iterate over batches
            for step, (x, y_true) in enumerate(train_ds):
                callback_list.on_batch_begin(step)
                with tf.GradientTape() as tape:
                    y_pred = self.model(x)
                    step_loss = self.criterion(y_true, y_pred)
                all_weights = list(itertools.chain.from_iterable(self.trainable_weights))
                grads = tape.gradient(step_loss, all_weights)
                for opt, weights in zip(self.optmizers, self.trainable_weights):
                    size = len(weights)
                    opt.apply_gradients(zip(grads[:size], weights))
                    grads = grads[size:]
                logs['loss'] = step_loss.numpy()
                pbar.update(step+1, [(k, logs[k]) for k in logs])
                callback_list.on_batch_begin(step, logs)
            callback_list.on_epoch_end(n, logs)
            if self.model.stop_training:
                break
        callback_list.on_train_end()
