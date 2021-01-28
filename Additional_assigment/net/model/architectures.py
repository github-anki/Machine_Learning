from net.layers import *
from net.layers.conv import Conv2D
from net.layers.dropout import Dropout
from net.layers.flatten import Reshape, Flatten
from net.layers.pooling import MaxPool2D
from net.model.model import Model


class MlpNet(Model):
    name = 'MlpNet'
    """
    simple architecture for flatten neural network with only dense hidden layers
    """

    def __init__(self, optimizer=None, initializer=None, metrics=None, loss_fun=None,
                 activation=None, hidden_units=None):
        Model.__init__(self, optimizer, metrics, loss_fun)
        self.activation = activation
        self.hidden_size = hidden_units

        self.add(Input(shape=(None, 28 * 28)))
        for units in hidden_units:
            self.add(Dense(units=units, weights_initializer=initializer))
            self.add(activation())
        self.add(Dense(units=10))

class ConvNet(Model):
    name = 'ConvNet'

    def __init__(self, optimizer=None, initializer=None, metrics=None, loss_fun=None,
                 activation=ReLU, kernel_size=3, filters=8, stride=1, padding=0, dropout=False):
        Model.__init__(self, optimizer, metrics, loss_fun)

        self.add(Input(shape=(None, 28 * 28)))
        self.add(Reshape(output_shape=(None, 28, 28, 1)))
        # N x 26 x 26 x 8
        self.add(
            Conv2D(filters=filters, kernel_size=kernel_size, stride=stride, padding=padding,
                   kernel_initializer=initializer))
        self.add(activation())
        self.add(MaxPool2D())  # N x 13 x 13 x 8
        if dropout:
            self.add(Dropout(drop_rate=0.25))
        self.add(Flatten())
        self.add(Dense(units=100, weights_initializer=initializer))
        self.add(activation())
        if dropout:
            self.add(Dropout(drop_rate=0.6))
        self.add(Dense(units=10, weights_initializer=initializer))

