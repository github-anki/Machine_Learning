import numpy as np
from sklearn.metrics import confusion_matrix


class Metric(object):

    def __call__(self, y_pred, y_true, **kwargs):
        raise NotImplementedError


class LabelAccuracy(Metric):
    name = 'label_accuracy'

    def __call__(self, y_pred, y_true, **kwargs):
        labels_amount = len(np.unique(y_true, return_counts=True)[1])
        labels = np.arange(labels_amount)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return np.diag(cm).sum() / cm.sum()
