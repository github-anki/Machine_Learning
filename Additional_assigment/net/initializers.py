import numpy as np

from net import dtype


class Initializer(object):
    """
    Base class for weights initializer
    """

    def __str__(self):
        return type(self).__name__

    def __call__(self, shape):
        """
        Inits weights for layers
        shape: shape of weights
        :return: weights initial values
        """
        raise NotImplementedError


class RandomInit(Initializer):
    name = 'random init'
    """
        only for Dense layer
    """
    act_fun = ['sigmoid', 'relu']

    def __init__(self, activation_fun):
        super().__init__()
        self._activation_fun = activation_fun

    def __call__(self, shape):
        fan_in, _ = get_fans(shape)
        a = None
        if self._activation_fun == 'sigmoid':
            a = 2.38
        elif self._activation_fun == 'relu':
            a = 2.0
        elif self._activation_fun == 'tanh':
            a = 1.0
        else:
            raise ValueError('Missing specified activation function')

        s = a / np.sqrt(fan_in)
        W = np.random.uniform(-s, s, size=shape).astype(dtype())
        return W


class RangeInit(Initializer):
    name = 'range init'

    def __init__(self, range):
        self._range = range

    def __call__(self, shape):
        low, high = self._range
        W = np.random.uniform(low, high, shape).astype(dtype())
        return W


class Randomization(Initializer):
    name = 'randomization init'
    """
    only for Dense layer
    """

    def __call__(self, shape):
        fan_in, fan_out = get_fans(shape)
        s = fan_out ** (1.0 / fan_in)
        W = np.random.uniform(-s, s, size=shape).astype(dtype())
        return W


class KaimingHe(Initializer):
    name = 'kaiming he'
    """
    initializer for ReLu activation function
    """

    def __call__(self, shape):
        fan_in, fan_out = get_fans(shape)
        scale = np.sqrt(2.0 / fan_in)
        W = (np.random.randn(*shape) * scale).astype(dtype())
        return W


class Xavier(Initializer):
    name = 'xavier'

    def __call__(self, shape):
        fan_in, fan_out = get_fans(shape)
        s = np.sqrt(6 / (fan_in + fan_out))
        W = np.random.uniform(-s, s, size=shape).astype(dtype())
        return W


class ZeroInit(Initializer):
    name = 'zero init'
    """
    Inits all weight to zero
    """

    def __call__(self, shape):
        W = np.zeros(shape, dtype=dtype())
        return W


class NormalInit(Initializer):
    name = 'normal init'

    def __call__(self, shape):
        W = np.random.normal(size=shape).astype(dtype())
        return W


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out
