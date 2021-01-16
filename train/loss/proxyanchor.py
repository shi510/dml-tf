import tensorflow as tf


class ProxyAnchorLoss(tf.keras.losses.Loss):

    def __init__(self, n_embedding, n_class, scale=30, delta=0.1, **kwargs):
        super(ProxyAnchorLoss, self).__init__(**kwargs)
        self.n_class = n_class
        self.scale = scale
        self.delta = delta
        self.initializer = tf.keras.initializers.Orthogonal()
        self.proxies = tf.Variable(name='proxies',
            initial_value=self.initializer((self.n_class, n_embedding)),
            trainable=True)
        self.trainable_weights = [self.proxies]


    def call(self, y_true, y_pred):
        onehot = tf.one_hot(y_true, self.n_class, 1., 0.)
        n_positives = tf.math.reduce_sum(onehot)
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_p = tf.math.l2_normalize(self.proxies, axis=1)
        dist = tf.matmul(norm_x, tf.transpose(norm_p))
        pos = -1 * self.scale * (dist - self.delta)
        neg = self.scale * (dist + self.delta)
        # select a distance between example and positive proxy.
        pos = tf.where(onehot == 1., pos, 0.)
        pos = tf.math.reduce_logsumexp(pos, axis=0)
        pos = tf.math.reduce_sum(tf.math.softplus(pos)) / n_positives
        # select all distance summation between example and negative proxy.
        neg = tf.where(onehot == 1., 0., neg)
        neg = tf.math.reduce_logsumexp(neg, axis=0)
        neg = tf.math.reduce_sum(tf.math.softplus(neg)) / self.n_class
        return pos + neg
