import tensorflow as tf

import evalutate.recall as recall


class LogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
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

    def __init__(self, test_ds, top_k, metric, log_dir=None):
        self.ds = test_ds
        self.top_k = top_k
        self.metric = metric
        if log_dir is not None:
            self.log_dir = log_dir
            self.writer = tf.summary.create_file_writer(self.log_dir)


    def on_train_end(self, logs=None):
        self.writer.close()


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        recall_top_k = recall.evaluate(self.model, self.ds, self.metric, self.top_k, 256)
        if hasattr(self, 'log_dir'):
            with self.writer.as_default():
                for name, value in zip(self.top_k, recall_top_k):
                    name = 'recall@' + str(name)
                    value *= 100
                    tf.summary.scalar(str(name), value, step=epoch)
                    logs[name] = value
                    print('{}: {}'.format(name, value))
                self.writer.flush()
