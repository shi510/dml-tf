import tensorflow as tf


@tf.function
def where_logsumexp(condition, x, axis):
    """
    numerical stable logsumexp applied to true condition only.
    logsumexp(x) = c + log(sum(exp(x-c))), where c is a maximum on the axis.
    """
    filtered = tf.where(condition, x, tf.zeros_like(x))
    x_max = tf.math.reduce_max(filtered, axis=axis)
    exp = tf.where(condition, tf.math.exp(x - x_max), tf.zeros_like(x))
    sumexp = tf.math.reduce_sum(exp, axis=axis)
    logsumexp = tf.where(sumexp > 0.,
        tf.math.log(sumexp) + x_max, tf.zeros_like(sumexp))
    return logsumexp

@tf.function
def where_softplus_logsumexp(condition, x, axis):
    """
    numerical stable softplus_logsumexp applied to true condition only.
    softplus(logsumexp(x)) = softplus(c + log(sum(exp(x-c)))),
        where c is a maximum on the axis.
    """
    filtered = tf.where(condition, x, tf.zeros_like(x))
    x_max = tf.math.reduce_max(filtered, axis=axis, keepdims=True)
    exp = tf.where(condition, tf.math.exp(x - x_max), tf.zeros_like(x))
    sumexp = tf.math.reduce_sum(exp, axis=axis)
    logsumexp = tf.where(sumexp > 0.,
        tf.math.softplus(tf.math.log(sumexp) + x_max),
        tf.zeros_like(sumexp))
    return logsumexp

@tf.function
def where_log1p_sumexp(condition, x, axis):
    """
    numerical stable log1p_sumexp applied to true condition only.
    log(1+sum(exp(x))) = c + log(exp(-c) + sum(exp(x-c))),
        where c is a maximum on the axis.
    """
    filtered = tf.where(condition, x, tf.zeros_like(x))
    x_max = tf.math.reduce_max(filtered, axis=axis)
    exp = tf.where(condition, tf.math.exp(x - x_max), tf.zeros_like(x))
    sumexp = tf.math.reduce_sum(exp, axis=axis)
    logsumexp = tf.where(sumexp > 0.,
        x_max + tf.math.log(tf.math.exp(-x_max) + sumexp),
        tf.zeros_like(sumexp))
    return logsumexp

class ProxyAnchorLoss:

    def __init__(self, n_embedding, n_class, scale=32, delta=0.1):
        self.n_class = n_class
        self.scale = scale
        self.delta = delta
        self.initializer = tf.keras.initializers.HeNormal()
        self.proxies = tf.Variable(name='proxies',
            initial_value=self.initializer([n_class, n_embedding]),
            trainable=True)
        self.trainable_weights = [self.proxies]

    def __call__(self, y_true, y_pred):
        """
        It converges very slowly or not when softplus is applied.
        Also a log1p(sumexp) very fluctuates positive loss.
        This implementation uses logsumexp.
        """
        pos_onehot = tf.one_hot(y_true, self.n_class, True, False)
        neg_onehot = tf.math.logical_not(pos_onehot)
        num_pos = tf.math.reduce_sum(tf.cast(pos_onehot, tf.float32))
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_p = tf.math.l2_normalize(self.proxies, axis=1)
        dist = tf.matmul(norm_x, norm_p, transpose_b=True)
        # positive examples of a proxy.
        pos = self.scale * (dist - self.delta)
        pos = where_logsumexp(pos_onehot, pos, axis=0)
        pos = tf.math.reduce_sum(pos) / num_pos
        # negative examples of a proxy.
        neg = self.scale * (dist + self.delta)
        neg = where_logsumexp(neg_onehot, neg, axis=0)
        neg = tf.math.reduce_mean(neg)
        # negative log: -log(a/b) = log(b) - log(a),
        #   where a is sum(exp(positivies)) and b is sum(exp(negatives)).
        # So minimizing this results in negative value.
        loss = neg - pos
        return loss, pos, neg
