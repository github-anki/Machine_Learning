import numpy as np

from net import learning_phase
from net.layers.layer import Layer


class Dropout(Layer):
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self._drop_rate = drop_rate
        self._mask = None

    def forward(self, x):
        if learning_phase():
            self._mask = np.random.rand(*x.shape) < (1 - self._drop_rate)
            # correct sum of all values according to expected EX considering drop rate
            return (x * self._mask) / (1 - self._drop_rate)
        else:
            return x

    def backward(self, x, dy):
        if learning_phase():
            return dict(dx=self._mask * dy)
        else:
            return dict(dx=dy)
