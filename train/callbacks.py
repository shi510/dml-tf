import os

import tensorflow as tf

import evalutate.recall as recall
import evalutate.nmi as nmi
from train.utils import plot_confusion_matrix
from train.utils import plot_to_image


class LogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir):
        self.log_dir = os.path.join(log_dir, 'scalar')
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=epoch)
            self.writer.flush()


class RecallCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_ds, top_k, metric, log_dir):
        self.ds = test_ds
        self.top_k = top_k
        self.metric = metric
        self.log_dir = os.path.join(log_dir, 'scalar')
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        recall_top_k = recall.evaluate(self.model, self.ds, self.metric, self.top_k, 256)
        with self.writer.as_default():
            for name, value in zip(self.top_k, recall_top_k):
                name = 'recall@' + str(name)
                value *= 100
                tf.summary.scalar(str(name), value, step=epoch)
                logs[name] = value
                print('{}: {}'.format(name, value))
            self.writer.flush()

class NMICallback(tf.keras.callbacks.Callback):

    def __init__(self, test_ds, classes, log_dir=None):
        self.ds = test_ds
        self.log_dir = log_dir
        self.classes = classes

    def on_train_end(self, logs=None):
        nmi_val = nmi.evaluate(
            self.model, self.ds, self.model.output.shape[-1], self.classes)
        nmi_val *= 100
        print('NMI: {}%'.format(nmi_val))
        if self.log_dir is not None:
            writer = tf.summary.create_file_writer(self.log_dir)
            with writer.as_default():
                tf.summary.scalar("NMI", nmi_val, step=0)
            writer.close()

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):

    def __init__(self, vectors, log_dir):
        self.log_dir = os.path.join(log_dir, 'image')
        self.vectors = vectors
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        # Use the model to predict the values from the validation dataset.

        # Calculate the confusion matrix.
        cm = tf.matmul(self.vectors, tf.transpose(self.vectors))
        cm = cm[:100, :100]
        min_val = tf.math.reduce_min(cm)
        max_val = tf.math.reduce_max(cm)
        cm = (cm - min_val) / (max_val - min_val)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm.numpy())
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
