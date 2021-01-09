from train.loss.utils import pairwise_distance

import tensorflow as tf

"""
All the code below is referenced from :
https://github.com/dichotomies/proxy-nca
"""

class ProxyNCALoss(tf.keras.losses.Loss):

    def __init__(self, n_embedding, n_class, scale_x=1, scale_p=3, **kwargs):
        super(ProxyNCALoss, self).__init__(**kwargs)
        self.n_class = n_class
        self.scale_x = scale_x
        self.scale_p = scale_p
        # Training convergence is sometimes slow, starting with low recall rate.
        #  - It may be due to initialization of proxy vectors.
        #  - It is better to use orthogonal initializer than random normal initializer.
        self.initializer = tf.keras.initializers.Orthogonal()
        self.proxies = tf.Variable(name='proxies',
            initial_value=self.initializer((self.n_class, n_embedding)),
            trainable=True)
        self.trainable_weights = [self.proxies]


    def call(self, y_true, y_pred):
        onehot = tf.one_hot(y_true, self.n_class, True, False)
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_p = tf.math.l2_normalize(self.proxies, axis=1)
        norm_x = norm_x * self.scale_x
        norm_p = norm_p * self.scale_p
        dist = pairwise_distance(norm_x, norm_p)
        dist = -1 * tf.maximum(dist, 0.)
        # for numerical stability,
        # all distances is substracted by its maximum value before exponentiating.
        dist = dist - tf.math.reduce_max(dist, axis=1, keepdims=True)
        # select a distance between example and positive proxy.
        pos = tf.where(onehot, dist, 0)
        pos = tf.math.reduce_sum(pos, axis=1)
        # select all distance summation between example and negative proxy.
        neg = tf.where(onehot, 0, tf.math.exp(dist))
        neg = tf.math.reduce_sum(neg, axis=1)
        # negative log_softmax: log(exp(a)/sum(exp(b)))=a-log(sum(exp(b)))
        loss = -1 * (pos - tf.math.log(neg))
        loss = tf.math.reduce_mean(loss)
        return loss
