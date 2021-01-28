import numpy as np

from net import dtype
from net.initializers import Xavier, ZeroInit
from net.layers.layer import Layer


class Conv2D(Layer):
    name = 'convolution2d'
    """
    Convolution layer performing the following operation:
      y = xW + b
      x - input of shape (N, H, W, D) assuming channel-last convention
            N - batch dimension
            H - height
            W - width
            D - depth (grey_scale=1, rgb=3) / channel dimension
      F - matrix of filters
      S - stride (sliding the filter)
      P - size of zero padding
      bias - bias of shape (filters,)
      y - output of shape (N, outH, outW, filters)
    """

    def __init__(self, filters, kernel_size, padding=0, stride=1, use_bias=True,
                 data_format='channels-last', kernel_initializer=Xavier(),
                 bias_initializer=ZeroInit()):
        super().__init__()
        self._filters = filters
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
        self._use_bias = use_bias
        self._data_format = data_format
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self._F = None
        self._bias = None

    def _build(self):
        """
        Creates layer parameters/variables
        F - matrix of shape (filters amount, kernel_size, kernel_size, input depth)
        bias of shape (filters amount,)
        """
        valid_hyperparams = self._check_hyperparams_for_valid_output()
        if not valid_hyperparams:
            raise ValueError("Wrong set of hyperparams. Output dimensions must be integer.")

        F_shape = (self._filters, self._kernel_size, self._kernel_size, self._input_shape[3])
        self._F = self.kernel_initializer(F_shape)
        if self._use_bias:
            self._bias = self.bias_initializer((1, 1, 1, self._filters))

    def _check_hyperparams_for_valid_output(self):
        h, w = self._input_shape[1:3]
        out_h = ((h - self._kernel_size + 2 * self._padding) / self._stride) + 1
        out_w = ((w - self._kernel_size + 2 * self._padding) / self._stride) + 1
        return out_w.is_integer() and out_h.is_integer()

    def output_shape(self):
        out_depth = self._filters
        h, w = self._input_shape[1:3]
        out_h = int(((h - self._kernel_size + 2 * self._padding) / self._stride) + 1)
        out_w = int(((w - self._kernel_size + 2 * self._padding) / self._stride) + 1)
        return None, out_h, out_w, out_depth

    def get_variables(self):
        if self._use_bias:
            return dict(F=self._F, bias=self._bias)
        return dict(F=self._F)

    def load_variables(self, vars):
        assert self._F.shape == vars["F"].shape
        self._F = vars["F"]
        if self._use_bias:
            assert self._bias.shape == vars["bias"].shape
            self._bias = vars["bias"]

    def apply_gradients(self, grads):
        self._F += grads["dW"]
        if self._use_bias:
            self._bias += grads["db"]

    def forward(self, x):
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
        return np.einsum('fhwd,nHWhwd->nHWf', self._F,
                         x_expanded) + self._bias if self._use_bias else \
            np.einsum('fhwd,nHWhwd->nHWf', self._F, x_expanded)

    def backward(self, x, dy):
        p = self._padding
        x_padded = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant', constant_values=0)

        N, in_H, in_W, D = x_padded.shape
        sN, sH, sW, sD = x_padded.strides

        dy_pad_x = np.insert(dy, np.repeat(np.arange(1, dy.shape[1]), self._stride - 1), 0, axis=2)
        dy_pad_y = np.insert(dy_pad_x, np.repeat(np.arange(1, dy.shape[2]), self._stride - 1), 0,
                             axis=1)

        # calculate dW as convolution of x and dy
        x_expanded = np.lib.stride_tricks.as_strided(x_padded,
                                                     shape=(
                                                         N,
                                                         self._kernel_size,
                                                         self._kernel_size,
                                                         D,
                                                         dy_pad_y.shape[1],
                                                         dy_pad_y.shape[2]),
                                                     strides=(
                                                         sN,
                                                         sH,
                                                         sW,
                                                         sD,
                                                         sH,
                                                         sW),
                                                     writeable=False).astype(dtype())
        dF = np.einsum('nkldhw,nhwf->fkld', x_expanded, dy_pad_y)

        dy_padded = np.pad(dy_pad_y, ((0, 0), (self._kernel_size - 1, self._kernel_size - 1),
                                      (self._kernel_size - 1, self._kernel_size - 1), (0, 0)),
                           mode='constant', constant_values=0)
        s_dy_N, s_dy_H, s_dy_W, s_dy_F = dy_padded.strides

        # calculate dx as convolution of dy_padded and rotated by 180 degrees F matrix
        dy_expanded = np.lib.stride_tricks.as_strided(dy_padded,
                                                      shape=(N,
                                                             in_H,
                                                             in_W,
                                                             self._filters,
                                                             self._kernel_size,
                                                             self._kernel_size),
                                                      strides=(sN,
                                                               s_dy_H,
                                                               s_dy_W,
                                                               s_dy_F,
                                                               dy_padded.strides[1],
                                                               dy_padded.strides[2]),
                                                      writeable=False).astype(dtype())
        dx = np.einsum('nHWfkl,fkld->nHWd', dy_expanded, np.rot90(self._F, 2, axes=(1, 2)))
        if self._padding != 0:
            dx = dx[:, self._padding:-self._padding, self._padding:-self._padding, :]

        if self._use_bias:
            db = np.sum(dy, axis=(0, 1, 2), keepdims=True)
            return dict(dx=dx, dW=dF, db=db)
        return dict(dx=dx, dW=dF)
