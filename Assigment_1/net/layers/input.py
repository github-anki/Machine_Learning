from net.layers.layer import Layer


class Input(Layer):
    def __init__(self, shape):
        self._output_shape = shape

    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        return x

    def backward(self, x, dy):
        return dict(dx=dy)
