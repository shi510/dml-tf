import tensorflow as tf


def softmax_loss(classes):
    def _func(true_y, predict_y):
        true_y = tf.one_hot(true_y, classes)
        loss = tf.nn.softmax_cross_entropy_with_logits(true_y, predict_y)
        return tf.math.reduce_mean(loss)
    return _func

def attach_linear(model, classes):
    model.trainable = False
    y = tf.keras.layers.Dense(classes)(model.output)
    return tf.keras.Model(model.input, y)


def evaluate(model, classes, train_ds, test_ds, opt):
    early_stop = tf.keras.callbacks.EarlyStopping(patience=3)
    model = attach_linear(model, classes)
    model.summary()
    model.compile(opt, softmax_loss(classes), 'sparse_categorical_accuracy')
    model.fit(train_ds, validation_data=test_ds, epochs=100, verbose=1,
        callbacks=[early_stop])
