import numpy as np


class Optimizer(object):
    def __init__(self, learning_rate=0.01):
        self._learning_rate = learning_rate

    def __str__(self):
        return type(self).__name__

    def compute_update(self, gradients):
        """
        Compute updates of trainable variables based on gradients computed in the backpropagation
        step.
        :param gradients: Dictionary containing layer names as keys and gradient dicts as values.
        """
        raise NotImplementedError

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate


class SGD(Optimizer):
    name = 'sgd'

    def __init__(self, learning_rate=0.01):
        Optimizer.__init__(self, learning_rate)

    def compute_update(self, gradients):
        for layer_name in gradients.keys():
            for var_name in gradients[layer_name].keys():
                gradients[layer_name][var_name] *= self._learning_rate
        return gradients


class SGDMomentum(Optimizer):
    name = 'sgd_momentum'

    def __init__(self, learning_rate=0.01, gamma=0.9):
        Optimizer.__init__(self, learning_rate)
        self._velocity = None
        self._gamma = gamma

    def _init(self, gradients):
        self._velocity = dict()

        for layer_name in gradients.keys():
            self._velocity[layer_name] = dict()
            self._velocity[layer_name] = {key: 0 for key, value in
                                          gradients[layer_name].items()}

    def compute_update(self, gradients):
        if self._velocity is None:
            self._init(gradients)

        for layer_name in gradients.keys():
            for var_name in gradients[layer_name].keys():
                v = self._velocity[layer_name][var_name]
                g = gradients[layer_name][var_name]
                self._velocity[layer_name][var_name] = self._gamma * v + self._learning_rate * g
        return self._velocity


class NAG(Optimizer):  # Nesterov Accelerated Gradient
    name = 'nesterov'

    def __init__(self, learning_rate=0.01, gamma=0.9):
        Optimizer.__init__(self, learning_rate)
        self._velocity = None
        self._gamma = gamma

    def _init(self, gradients):
        self._velocity = dict()
        for layer_name in gradients.keys():
            self._velocity[layer_name] = dict()
            self._velocity[layer_name] = {key: 0 for key, value in
                                          gradients[layer_name].items()}

    def compute_update(self, gradients):
        if self._velocity is None:
            self._init(gradients)

        for layer_name in gradients.keys():
            for var_name in gradients[layer_name].keys():
                v = self._velocity[layer_name][var_name]
                v_prev = v
                g = gradients[layer_name][var_name]
                v = self._gamma * v + self._learning_rate * g
                self._velocity[layer_name][var_name] = -self._gamma * v_prev + (
                        1 + self._gamma) * v
        return self._velocity


class Adagrad(Optimizer):
    name = 'adagrad'

    """
    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the learning rate.
    Denominator computes the L2 norm of all previous gradients on
    a per dimension basis and ETA is a global learning rate shared
    by all dimensions.

        eta - constant
        eps - usually set between [1e-4, 1e-8] to avoid division by zero
    """

    def __init__(self, learning_rate=0.01, eps=1e-6):
        Optimizer.__init__(self, learning_rate)
        self._eps = eps
        self._grad_acc = None

    def _init(self, gradients):
        self._grad_acc = dict()
        self._updates = dict()
        for layer_name in gradients.keys():
            self._grad_acc[layer_name] = dict()
            self._updates[layer_name] = dict()
            self._grad_acc[layer_name] = {key: 0 for key, value in
                                          gradients[layer_name].items()}
            self._updates[layer_name] = {key: 0 for key, value in
                                         gradients[layer_name].items()}

    def compute_update(self, gradients):
        if self._grad_acc is None:
            self._init(gradients)

        for layer_name in gradients.keys():
            for var_name in gradients[layer_name].keys():
                g = gradients[layer_name][var_name]

                # sum all squared gradient from the beginning
                self._grad_acc[layer_name][var_name] += g ** 2
                cache = self._grad_acc[layer_name][var_name]

                # calculate learning rate
                lr = self._learning_rate / (np.sqrt(cache) + self._eps)
                self._updates[layer_name][var_name] = lr * g

        return self._updates


class Adadelta(Optimizer):
    name = 'adadelta'

    """
    Adadelta is a more robust extension of Adagrad
    that adapts learning rates based on a moving window of gradient updates,
    instead of accumulating all past gradients. This way, Adadelta continues
    learning even when many updates have been done. Compared to Adagrad, in the
    original version of Adadelta you don't have to set an initial learning
    rate.

    Based on paper:
        rho - hyperparameter (best value 0.95)
    """

    def __init__(self, learning_rate=None, rho=0.95, eps=1e-6):
        Optimizer.__init__(self, learning_rate)
        self._rho = rho
        self._eps = eps
        self._grad_acc = None
        self._delta_acc = None

    def _init(self, gradients):
        # initialize of accumulation variables
        self._grad_acc = dict()
        self._delta_acc = dict()
        self._updates = dict()
        for layer_name in gradients.keys():
            self._grad_acc[layer_name] = {key: 0 for key, value in
                                          gradients[layer_name].items()}
            self._delta_acc[layer_name] = {key: 0 for key, value in
                                           gradients[layer_name].items()}
            self._updates[layer_name] = {key: 0 for key, value in
                                         gradients[layer_name].items()}

    def compute_update(self, gradients):
        if self._grad_acc is None or self._delta_acc is None:
            self._init(gradients)

        for layer_name in gradients.keys():
            for var_name in gradients[layer_name].keys():
                g = gradients[layer_name][var_name]
                delta_acc = self._delta_acc[layer_name][var_name]
                g_acc = self._grad_acc[layer_name][var_name]

                # update gradient accumulator
                g_acc_new = self._rho * g_acc + (1. - self._rho) * g ** 2
                self._grad_acc[layer_name][var_name] = g_acc_new

                # compute update using new grad accumulator and old update accumulator
                update = (np.sqrt(delta_acc + self._eps) / np.sqrt(g_acc_new + self._eps)) * g

                # update accumulator of updates
                delta_acc_new = self._rho * delta_acc + (1. - self._rho) * update ** 2
                self._delta_acc[layer_name][var_name] = delta_acc_new

                # return update
                self._updates[layer_name][var_name] = update

        return self._updates


class RMSprop(Optimizer):
    name = 'rmsprop'

    def __init__(self, learning_rate=None, eta=0.001, rho=0.9, eps=1e-6):
        Optimizer.__init__(self, learning_rate)
        self._rho = rho
        self._eps = eps
        self._eta = eta
        self._grad_acc = None

    def _init(self, gradients):
        self._grad_acc = dict()
        self._updates = dict()
        for layer_name in gradients.keys():
            self._grad_acc[layer_name] = {key: 0 for key, value in
                                          gradients[layer_name].items()}
            self._updates[layer_name] = {key: 0 for key, value in
                                         gradients[layer_name].items()}

    def compute_update(self, gradients):
        if self._grad_acc is None:
            self._init(gradients)

        for layer_name in gradients.keys():
            for var_name in gradients[layer_name].keys():
                g = gradients[layer_name][var_name]

                # calculate update of grad accumulator
                grad_acc_new = self._rho * self._grad_acc[layer_name][var_name] + (
                        1. - self._rho) * g ** 2
                self._grad_acc[layer_name][var_name] = grad_acc_new

                # calculate update of weights
                update = self._eta / np.sqrt(grad_acc_new + self._eps) * g
                self._updates[layer_name][var_name] = update

        return self._updates


class Adam(Optimizer):
    name = 'adam'
    """
    - Adaptive Moment Estimation -
    An adaptive learning rate algorithm for first-order gradient-based optimization of
    stochastic objective functions, based on adaptive estimates of lower-order moments.

    Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent
    with momentum. It uses the squared gradients to scale the learning rate like RMSprop
    and it takes advantage of momentum by using moving average of the gradient instead of
    gradient itself like SGD with momentum.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        Optimizer.__init__(self, learning_rate)
        self._eps = eps
        self._beta1 = beta1
        self._beta2 = beta2
        self._updates = None
        self._m = None
        self._v = None
        self._iter = 1

    def _init(self, gradients):
        self._updates = dict()
        self._m = dict()
        self._v = dict()
        for layer_name in gradients.keys():
            self._updates[layer_name] = dict()
            self._m[layer_name] = dict()
            self._v[layer_name] = dict()
            self._updates[layer_name] = {key: 0 for key, value in
                                         gradients[layer_name].items()}
            self._m[layer_name] = {key: 0 for key, value in
                                   gradients[layer_name].items()}
            self._v[layer_name] = {key: 0 for key, value in
                                   gradients[layer_name].items()}

    def compute_update(self, gradients):
        if self._m is None or self._v is None:
            self._init(gradients)

        for layer_name in gradients.keys():
            for var_name in gradients[layer_name].keys():
                g = gradients[layer_name][var_name]

                # calculate new m and v values
                m_new = self._beta1 * self._m[layer_name][var_name] + (1. - self._beta1) * g
                v_new = self._beta2 * self._v[layer_name][var_name] + (1. - self._beta2) * g ** 2

                # correct m and v values to ensure not going into 0 value
                m_hat = m_new / (1. - self._beta1 ** self._iter)
                v_hat = v_new / (1. - self._beta2 ** self._iter)

                self._m[layer_name][var_name] = m_new
                self._v[layer_name][var_name] = v_new

                # compute update for weights
                update = self._learning_rate / (np.sqrt(v_hat) + self._eps) * m_hat
                self._updates[layer_name][var_name] = update

        self._iter += 1
        return self._updates
