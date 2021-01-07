import tensorflow as tf

"""
All the code below is referenced from :
https://github.com/dichotomies/proxy-nca
"""

def pairwise_distance(A, B):
    """
    (a-b)^2 = a^2 -2ab + b^2
    A shape = (N, D)
    B shaep = (C, D)
    result shape = (N, C)
    """
    row_norms_A = tf.math.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.math.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


class ProxyNCALoss(tf.keras.losses.Loss):

    def __init__(self, embedding_dim, classes, scale_x=1, scale_p=3, **kwargs):
        super(ProxyNCALoss, self).__init__(**kwargs)
        self.classes = classes
        self.scale_x = scale_x
        self.scale_p = scale_p
        self.proxies = tf.Variable(name='proxies',
            initial_value=tf.random.normal((self.classes, embedding_dim)),
            trainable=True)


    def call(self, y_true, y_pred):
        binarised = tf.one_hot(y_true, self.classes, True, False)
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_proxies = tf.math.l2_normalize(self.proxies, axis=1)
        norm_x = norm_x * self.scale_x
        norm_proxies = norm_proxies * self.scale_p
        dist = pairwise_distance(norm_x, norm_proxies)
        # all distances should be greater than 0.
        dist = tf.maximum(dist, 0)
        dist = -1 * dist
        # for numerical stability,
        # all distances is substracted by its maximum value before exponentiating.
        dist = dist - tf.math.reduce_max(dist, axis=1, keepdims=True)
        dist = tf.math.exp(dist)
        # select a distance between example and positive proxy.
        pos = tf.math.reduce_sum(tf.where(binarised, dist, 0), axis=1)
        # select all distance summation between example and negative proxy.
        neg = tf.math.reduce_sum(tf.where(binarised, 0, dist), axis=1)
        # avoid dividing by zero.
        neg = tf.maximum(neg, 1e-6)
        # negative log_softmax.
        loss = -1 * tf.math.log(pos / neg)
        # select only positive loss.
        loss = tf.maximum(loss, 0)
        loss = tf.math.reduce_mean(loss)
        return loss
