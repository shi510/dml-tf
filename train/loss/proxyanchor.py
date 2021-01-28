import tensorflow as tf


@tf.function
def where_logsumexp(condition, x, axis):
    """
    numerical stable logsumexp
    logsumexp(x) = c + log(sum(x-c)), where c is a maximum along with the axis.
    """
    filtered = tf.stop_gradient(
        tf.where(condition, x, tf.zeros_like(x)))
    x_max = tf.stop_gradient(
        tf.math.reduce_max(filtered, axis=axis, keepdims=True))
    exp = tf.where(condition, tf.exp(x - x_max), tf.zeros_like(x))
    sumexp = tf.math.reduce_sum(exp, axis=axis, keepdims=True)
    logsumexp = tf.math.log(sumexp) + x_max
    return logsumexp

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
        pos_onehot = tf.one_hot(y_true, self.n_class, True, False)
        neg_onehot = tf.math.logical_not(pos_onehot)
        num_pos = tf.math.reduce_sum(tf.cast(pos_onehot, tf.float32))
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_p = tf.math.l2_normalize(self.proxies, axis=1)
        dist = tf.matmul(norm_x, tf.transpose(norm_p))
        # select all distances between example and positive proxy.
        pos = -1 * self.scale * (dist - self.delta)
        pos = where_logsumexp(pos_onehot, pos, axis=0)
        pos = tf.math.reduce_sum(tf.math.softplus(pos)) / num_pos
        # select all distances between example and negative proxy.
        neg = self.scale * (dist + self.delta)
        neg = where_logsumexp(neg_onehot, neg, axis=0)
        neg = tf.math.reduce_sum(tf.math.softplus(neg)) / self.n_class
        return pos + neg
