import numpy as np

class Batcher(object):
    def __init__(self, x, y):
        """
        :param x: array of an arbitrary shape with first dimension is used to code samples. (N, D)
        :param y: sample labels N
        """
        self._x = x
        self._y = y
        self._num_samples = len(y)
        self._init()
        self._epoch = 0

    def _init(self):
        self._permutation = np.random.permutation(self._num_samples)
        self._consumed = 0

    def __call__(self, size):
        if self._consumed + size <= self._num_samples:
            indices = self._permutation[self._consumed:self._consumed + size]
            x = self._x[indices]
            y = self._y[indices]
            self._consumed += size
        else:
            indices = self._permutation[self._consumed:]
            x = self._x[indices]
            y = self._y[indices]
            remaining = size - x.shape[0]
            self._init()
            rem_x, rem_y = self(remaining)
            x = np.concatenate([x, rem_x])
            y = np.concatenate([y, rem_y])
            self._epoch += 1
        return x, y

    def epoch(self):
        return self._epoch + float(self._consumed) / self._num_samples
