import numpy as np

from net.initializers import Xavier, ZeroInit
from net.layers.layer import Layer


class Dense(Layer):
    name = 'dense'
    """
    Dense layer performing the following operation:
      y = xW + b
      x - input of shape (N, M)
      W - weights of shape (M, units)
      b - bias of shape (units,)
      y - output of shape (N, units)
    """

    def __init__(self, units, weights_initializer=Xavier(),
                 bias_initializer=ZeroInit(), use_bias=True):
        super().__init__()
        self.units = units
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._use_bias = use_bias
        self._W = None
        self._bias = None

    def load_variables(self, vars):
        assert self._W.shape == vars["W"].shape
        self._W = vars["W"]
        if self._use_bias:
            assert self._bias.shape == vars["bias"].shape
            self._bias = vars["bias"]

    def get_variables(self):
        return dict(W=self._W, bias=self._bias) if self._use_bias else dict(W=self._W)

    def apply_gradients(self, grads):
        self._W += grads["dW"]
        if self._use_bias:
            self._bias += grads["db"]

    def _build(self):
        assert len(self._input_shape) == 2, "input array to dense layer should be two-dimensional"
        W_shape = self._input_shape[1], self.units
        self._W = self.weights_initializer(W_shape)

        if self._use_bias:
            self._bias = self.bias_initializer((1, self.units))

    def output_shape(self):
        return None, self.units

    def forward(self, x):
        return np.dot(x, self._W) + self._bias if self._use_bias else np.dot(x, self._W)

    def backward(self, x, dy):
        """
        :param x: input of shape (N, M)
        :param dy: upstream gradient of shape (N, units)
        :return: dict with gradients:
                  - dx - gradient w.r.t. input x of shape (N, M)
                  - dW - gradient w.r.t. weights W of shape (M, units)
                  - db - gradient w.r.t. bias b of shape (units, )
        """

        dx = np.dot(dy, self._W.T)
        dW = np.dot(x.T, dy)
        if self._use_bias:
            db = dy.sum(axis=0, keepdims=True)
            return dict(dx=dx, dW=dW, db=db)
        return dict(dx=dx, dW=dW)
