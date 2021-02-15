import tensorflow as tf

@tf.function
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

    dist = row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B
    return tf.math.maximum(dist, 0.)


def smooth_one_hot(labels, n_class, smooth_factor=0.1):
    pos = (1. - smooth_factor)
    neg = smooth_factor / (n_class - 1.)
    onehot = tf.one_hot(labels, n_class, pos, neg)
    return onehot

def orthogonality(mat):
    dim = tf.cast(tf.shape(mat)[0], tf.float32)
    self_dist = tf.matmul(mat, tf.transpose(mat))
    diag = tf.linalg.diag_part(self_dist)
    diag = tf.math.square(1. - diag)
    diag = tf.math.reduce_sum(diag) / dim
    upper_right = tf.linalg.band_part(self_dist, 0, -1)
    upper_right = tf.linalg.set_diag(upper_right, tf.zeros([dim]))
    upper_right = tf.math.square(upper_right)
    upper_right = tf.math.reduce_sum(upper_right) / ((dim - 1.) * 2.)
    loss = diag + upper_right
    return loss
