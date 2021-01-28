from collections import defaultdict


class LossAccLogger(object):
    def __init__(self, dirs=('loss_epoch', 'accuracy', 'loss_batch')):
        self._dirs = dirs
        self._tags = ['train', 'val']
        self.logging_data = defaultdict()
        self._init_logging_data()

    def _init_logging_data(self):
        for directory in self._dirs:
            self.logging_data[directory] = {tag: [] for tag in self._tags}

    def add_loss_per_epoch(self, tag, loss):
        self.logging_data['loss_epoch'][tag].append(loss)

    def add_accuracy_per_epoch(self, tag, accuracy):
        self.logging_data['accuracy'][tag].append(accuracy)

    def add_loss_per_batch(self, loss, tag='train'):
        self.logging_data['loss_batch'][tag].append(loss)
