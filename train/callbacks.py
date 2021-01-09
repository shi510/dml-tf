import os

import tensorflow as tf


class LogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir='./logs'):
        self.log_dir = os.path.join(log_dir, 'scalars')
        self.writer = tf.summary.create_file_writer(self.log_dir)


    def on_train_end(self, logs=None):
        self.writer.close()


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=epoch)
        self.writer.flush()
