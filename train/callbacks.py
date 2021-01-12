import os

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import backend as K
import numpy as np

import evalutate.recall as recall


class LogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)


    def on_train_end(self, logs=None):
        self.writer.close()


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=epoch)
        self.writer.flush()


class RecallCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_ds, top_k, log_dir=None):
        self.ds = test_ds
        self.top_k = top_k
        if log_dir is not None:
            self.log_dir = log_dir
            self.writer = tf.summary.create_file_writer(self.log_dir)


    def on_train_end(self, logs=None):
        self.writer.close()


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        recall_top_k = recall.evaluate(self.model, self.ds, self.top_k, 256)
        if hasattr(self, 'log_dir'):
            with self.writer.as_default():
                for name, value in zip(self.top_k, recall_top_k):
                    name = 'recall@' + str(name)
                    value *= 100
                    tf.summary.scalar(str(name), value, step=epoch)
                    logs[str(name)] = value
            self.writer.flush()

class ReduceLROnPlateau(tf.keras.callbacks.Callback):

    def __init__(self,
                 optimizer,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 **kwargs):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                            'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.optimizer = optimizer
        self._reset()


    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
            (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['proxy_lr'] = float(K.get_value(self.optimizer.lr))
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0