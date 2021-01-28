from net.initializers import *
from net.layers.activations import *
from net.losses import categorical_cross_entropy
from net.metrics import LabelAccuracy
from net.optimizers import *
from .architectures import *


def get_model(model_name, optimizer='SGDMomentum', initializer='Xavier', activation='sigmoid',
              loss_fun='crossEntropy', hidden_units=(512,)):
    metrics = [LabelAccuracy()]
    optimizer = get_optimizer(optimizer)
    initializer = get_initializer(initializer, activation)
    loss_fun = get_loss_function(loss_fun)
    activation = get_activation(activation)

    models = dict(
        MlpNet=MlpNet(optimizer, initializer, metrics, loss_fun, activation, hidden_units)
    )

    m = models[model_name]
    print("Creating model {} with parameters count: {}".format(model_name, m.param_count()))
    return m


def get_initializer(initializer_name, activation):
    initializers = dict(
        RandomInit=RandomInit(activation),
        Randomization=Randomization(),
        Xavier=Xavier(),
        KaimingHe=KaimingHe(),
        Zero=ZeroInit(),
        Normal=NormalInit()
    )

    initializer = initializers[initializer_name]
    print("Using {} initializer".format(initializer.name))
    return initializer


def get_loss_function(loss_fun_name):
    loss_functions = dict(
        crossEntropy=categorical_cross_entropy
    )
    loss_fun = loss_functions[loss_fun_name]
    print("Using {} loss function".format(loss_fun_name))
    return loss_fun


def get_optimizer(optimizer_name):
    optimizers = dict(
        SGD=SGD(),
        SGDMomentum=SGDMomentum(),
        NAG=NAG(),
        Adagrad=Adagrad(),
        Adadelta=Adadelta(),
        RMSprop=RMSprop(),
        Adam=Adam(),
    )

    optimizer = optimizers[optimizer_name]
    print("Using {} optimizer".format(optimizer.name))
    return optimizer


def get_activation(activation_name):
    activations = dict(
        relu=ReLU,
        sigmoid=Sigmoid,
        tanh=Tanh
    )
    activation = activations[activation_name]
    print("Using {} activation function in hidden layers".format(activation.name))
    return activation
