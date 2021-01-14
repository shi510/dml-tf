from train.loss.utils import pairwise_distance

import tensorflow as tf

"""
All the code below is referenced from :
https://github.com/euwern/proxynca_pp
"""

class ProxyNCAPlusLoss(tf.keras.losses.Loss):

    def __init__(self, n_embedding, n_class, temperature_scale=1./9., **kwargs):
        super(ProxyNCAPlusLoss, self).__init__(**kwargs)
        self.n_class = n_class
        self.temperature_scale = 1. / temperature_scale
        # Training convergence is sometimes slow, starting with low recall rate.
        #  - It may be due to initialization of proxy vectors.
        #  - It is better to use orthogonal initializer than random normal initializer.
        self.initializer = tf.keras.initializers.Orthogonal()
        self.proxies = tf.Variable(name='proxies',
            initial_value=self.initializer((self.n_class, n_embedding)),
            trainable=True)
        self.trainable_weights = [self.proxies]


    def call(self, y_true, y_pred):
        onehot = tf.one_hot(y_true, self.n_class, 1., 0.)
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_p = tf.math.l2_normalize(self.proxies, axis=1)
        dist = pairwise_distance(norm_x, norm_p) * self.temperature_scale
        dist = -1 * tf.maximum(dist, 0.)
        loss = -1 * onehot * tf.math.log_softmax(dist, axis=1)
        loss = tf.math.reduce_sum(loss, axis=1)
        loss = tf.math.reduce_mean(loss)
        return loss
