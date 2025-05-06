"""
Custom callbacks for model training.
"""

import tensorflow as tf
import keras
import time


class TimeHistory(keras.callbacks.Callback):
    """
    Callback to track training time for each epoch
    """

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.times = []
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch_start_time is not None:
            logs['time'] = time.time() - self.epoch_start_time
            self.times.append(logs['time'])


class CustomTensorBoard(keras.callbacks.TensorBoard):
    """
    Custom TensorBoard callback with additional metrics
    """

    def __init__(self, log_dir='./logs', **kwargs):
        super(CustomTensorBoard, self).__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Calculate additional metrics if precision and recall are available
        if 'precision' in logs and 'recall' in logs and logs['precision'] > 0 and logs['recall'] > 0:
            logs['f1_score'] = 2 * ((logs['precision'] * logs['recall']) /
                                    (logs['precision'] + logs['recall']))

        super(CustomTensorBoard, self).on_epoch_end(epoch, logs)