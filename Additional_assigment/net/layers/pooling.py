from net import dtype
from net.layers.layer import Layer
import numpy as np


class MaxPool2D(Layer):
    """
    Max Pooling layer reduces dimensionality of the input data:
      y = xW + b
      x - input of shape (N, H, W, D)
            N - batch dimension
            W - width
            H - height
            D - depth / channel dimension
      F - matrix of filters
      S - stride (sliding the filter)
      y - output of shape (N, H/2, W/2, D)
    """

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._mask = None

    def _build(self):
        pass

    def output_shape(self):
        """
        Input shape - N, H, W, D
        Output shape - N, H/2, W/2, D
        """
        out_depth = self._input_shape[-1]
        height, width = self._input_shape[1:3]

        # height/2 and width/2
        out_h = int(((height - self._kernel_size + 2 * self._padding) / self._stride) + 1)
        out_w = int(((width - self._kernel_size + 2 * self._padding) / self._stride) + 1)
        return None, out_h, out_w, out_depth

    def _expand_input(self, x):
        p = self._padding
        x_padded = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant', constant_values=0)
        N, H, W, D = x_padded.shape
        sN, sH, sW, sD = x_padded.strides
        _, out_h, out_w, _ = self.output_shape()

        x_expanded = np.lib.stride_tricks.as_strided(x_padded,
                                                     shape=(N, out_h, out_w,
                                                            self._kernel_size,
                                                            self._kernel_size, D),
                                                     strides=(
                                                         sN,
                                                         sH * self._stride,
                                                         sW * self._stride,
                                                         sH, sW, sD),
                                                     writeable=False).astype(dtype())
        return x_expanded

    def forward(self, x):
        x_expanded = self._expand_input(x)
        max_values = np.max(x_expanded, axis=(-3, -2), keepdims=True)
        self._mask = (x_expanded == max_values)
        return np.squeeze(max_values, axis=(-3, -2))

    def backward(self, x, dy):
        dy_n, dy_h, dy_w, dy_d = dy.shape
        dy_masked = self._mask * dy.reshape((dy_n, dy_h, dy_w, 1, 1, dy_d))
        dx = np.zeros_like(x)

        _, out_h, out_w, out_d = self.output_shape()
        N, H, W, D = x.shape
        sN, sH, sW, sD = x.strides
        expanded_dx = np.lib.stride_tricks.as_strided(dx,
                                                      shape=(N, out_h, out_w,
                                                             self._kernel_size,
                                                             self._kernel_size, D),
                                                      strides=(sN,
                                                               sH * self._stride,
                                                               sW * self._stride,
                                                               sH, sW, sD),
                                                      writeable=True)
        np.add.at(expanded_dx, (), dy_masked)
        shape = expanded_dx.shape
        dx = np.reshape(expanded_dx, (shape[0], shape[1] * shape[3], shape[2] * shape[4], shape[5]))
        return dict(dx=dx)
