from train.loss.proxynca import ProxyNCALoss
from train.loss.proxyanchor import ProxyAnchorLoss

import tensorflow as tf
import tensorflow_addons as tfa


class ProxyNCAModel(tf.keras.Model):

    def __init__(self, n_embeddings, n_classes, proxy_lr=1e-3, scale=32, **kwargs):
        super(ProxyNCAModel, self).__init__(**kwargs)
        self.n_embeddings = n_embeddings
        self.n_classes = n_classes
        self.proxy_lr = proxy_lr
        self.proxy_loss = ProxyNCALoss(n_embeddings, n_classes, scale)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    def compile(self, optimizer, **kwargs):
        super(ProxyNCAModel, self).compile(**kwargs)
        self.optimizer = optimizer
        self.optimizer_proxy = tfa.optimizers.AdamW(learning_rate=self.proxy_lr, weight_decay=1e-4)

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.proxy_loss(y_true, y_pred)
        trainable_vars = self.trainable_weights
        trainable_vars += self.proxy_loss.trainable_weights
        grads = tape.gradient(loss, trainable_vars)
        spliter = len(trainable_vars) - len(self.proxy_loss.trainable_weights)
        self.optimizer.apply_gradients(zip(grads[:spliter], trainable_vars[:spliter]))
        self.optimizer_proxy.apply_gradients(zip(grads[spliter:], trainable_vars[spliter:]))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

class ProxyAnchorModel(tf.keras.Model):

    def __init__(self, n_embeddings, n_classes, proxy_lr, scale, delta, **kwargs):
        super(ProxyAnchorModel, self).__init__(**kwargs)
        self.n_embeddings = n_embeddings
        self.n_classes = n_classes
        self.proxy_lr = proxy_lr
        self.proxy_loss = ProxyAnchorLoss(n_embeddings, n_classes, scale, delta)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    def compile(self, optimizer, **kwargs):
        super(ProxyAnchorModel, self).compile(**kwargs)
        self.optimizer = optimizer
        self.optimizer_proxy = tfa.optimizers.AdamW(learning_rate=self.proxy_lr, weight_decay=1e-4)

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.proxy_loss(y_true, y_pred)
        trainable_vars = self.trainable_weights
        trainable_vars += self.proxy_loss.trainable_weights
        grads = tape.gradient(loss, trainable_vars)
        spliter = len(trainable_vars) - len(self.proxy_loss.trainable_weights)
        self.optimizer.apply_gradients(zip(grads[:spliter], trainable_vars[:spliter]))
        self.optimizer_proxy.apply_gradients(zip(grads[spliter:], trainable_vars[spliter:]))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]
