import numpy as np

from net.layers.layer import Layer


class Flatten(Layer):

    def __init__(self):
        super().__init__()

    def output_shape(self):
        return self._input_shape[0], np.prod(self._input_shape[1:])

    def forward(self, x):
        return np.reshape(x, (-1, np.prod(self._input_shape[1:])))

    def backward(self, x, dy):
        return dict(dx=np.reshape(dy, (-1,) + x.shape[1:]))


class Reshape(Layer):

    def __init__(self, output_shape):
        super().__init__()
        self._output_shape = output_shape

    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        return x.reshape((-1,) + self._output_shape[1:])

    def backward(self, x, dy):
        return dict(dx=np.reshape(dy, (-1,) + self._input_shape[1:]))
