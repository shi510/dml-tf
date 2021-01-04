import tensorflow as tf

"""
All the code below is referenced from :
https://github.com/dichotomies/proxy-nca
"""

def pairwise_distance(A, B):
    row_norms_A = tf.math.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.math.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


def binarize_and_smooth_labels(labels, classes, smoothing_const = 0.1):
    binarised = tf.one_hot(labels, classes)
    pos_const = 1.0 - smoothing_const
    neg_const = smoothing_const / (classes - 1.0)
    return tf.where(binarised == 1, pos_const, neg_const)



def ProxyNCALoss(classes):
    """
    pos_dist = dist(x, p(y))
    neg_dist = sum(x, p(z))
    loss = -log(exp(pos_dist) / sum(exp(neg_dist)))
    """
    def _loss_fn(labels, probs):
        # smooth_labels = binarize_and_smooth_labels(labels, classes)
        # loss = tf.math.reduce_mean(tf.multiply(smooth_labels, probs))
        binarised = tf.one_hot(labels, classes)
        probs = tf.math.exp(-1 * probs)
        pos = tf.math.reduce_sum(tf.where(binarised == 1, probs, 0), axis=1)
        neg = tf.math.reduce_sum(tf.where(binarised == 0, probs, 0), axis=1)
        loss = -1 * tf.math.log(pos / neg)
        loss = tf.where(loss > 0, loss, 0)
        loss = tf.math.reduce_mean(loss)
        return loss
    return _loss_fn


class ProxyNCALayer(tf.keras.layers.Layer):

    def __init__(self, embedding_dim, classes, scale_x=1, scale_p=3, **kwargs):
        super(ProxyNCALayer, self).__init__(**kwargs)
        self.classes = classes
        self.scale_x = scale_x
        self.scale_p = scale_p


    def build(self, input_shape):
        self.proxies = self.add_weight(name='proxies', 
            shape=(self.classes, input_shape[-1]),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.5),
            trainable=True)


    def call(self, embeddings):
        # print(embeddings[0][:5])
        norm_x = tf.math.l2_normalize(embeddings, axis=1)
        norm_proxies = tf.math.l2_normalize(self.proxies, axis=1)
        norm_x = norm_x * self.scale_x
        norm_proxies = norm_proxies * self.scale_p
        dist = pairwise_distance(norm_x, norm_proxies)
        # dist = tf.math.square(dist)
        # probs = -1 * tf.nn.log_softmax(-1*dist, axis=1)
        return dist


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.classes)
