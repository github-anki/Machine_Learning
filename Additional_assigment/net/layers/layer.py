class Layer(object):
    """
    Base class for neural network layers.
    """

    def __init__(self):
        self._input_shape = None
        self._initializer = None
        self._optimizer = None

    def __str__(self):
        return type(self).__name__

    def output_shape(self):
        return self._input_shape

    def build(self, input_shape, optimizer=None):
        """
        Initialize layer variables for given input shape.
        :param optimizer: optimizer object
        :param input_shape: tuple with first entry corresponding to batch dimension
        """
        self._input_shape = input_shape
        self._optimizer = optimizer
        self._build()

    def _build(self):
        pass

    def load_variables(self, vars):
        """
        Load layer trainable variables.
        :param vars: Dictionary containing trainable variables.
        """
        pass

    def get_variables(self):
        """
        Return layer trainable variables as a dictionary.
        """
        return dict()

    def apply_gradients(self, grads):
        """
        Apply gradients to layer trainable variables.
        """
        pass

    def forward(self, x):
        """
        Perform forward pass.
        :param x: Input array.
        :return:  Output array.
        """
        raise NotImplementedError

    def backward(self, x, dy):
        """
        Perform backward pass (backpropagation).
        :param x: Input array.
        :param dy: Upstream gradient.
        :return: Dictionary with gradients w.r.t. to the input and trainable variables.
        """
        raise NotImplementedError
