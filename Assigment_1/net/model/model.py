import os
import pickle
from collections import defaultdict

from net import *
from net.losses import softmax, categorical_cross_entropy
from net.optimizers import SGDMomentum, SGD


class Model(object):
    def __init__(self, optimizer=None, metrics=None, loss_fun=None):
        self._layers = []
        self._optimizer = optimizer if optimizer is not None else SGD()
        self._metrics = metrics
        self._loss_func = loss_fun if loss_fun is not None else categorical_cross_entropy

    def add(self, layer):
        if len(self._layers) > 0:
            layer.build(self._layers[-1].output_shape(), self._optimizer)
        self._layers.append(layer)

    def setup_train(self, learning_rate):
        # setup optimizer
        if self._optimizer is not None:
            self._optimizer.set_learning_rate(learning_rate)
        else:
            self._optimizer = SGDMomentum(learning_rate=learning_rate)

    def train(self, x, y):
        set_learning_phase(True)

        x, cache = self.forward_pass(x)
        loss, dscores = self.compute_loss(x, y)
        updates = self.backward_pass(dscores, cache)

        updates = self._optimizer.compute_update(updates)
        for idx, l in enumerate(self._layers):
            l.apply_gradients(updates["layer{}".format(idx)])

        return loss

    def forward_pass(self, x, learning_phase=True):
        set_learning_phase(learning_phase)
        cache = []
        for l in self._layers:
            cache.append(x)
            x = l.forward(x)
        return x, cache

    def backward_pass(self, dscores, x_cache):
        gradients = {}
        dy = dscores
        for idx in reversed(range(len(self._layers))):
            # g - dict of grad values from each layer
            x = x_cache[idx]

            g = self._layers[idx].backward(x, dy)
            # dx from prev layer is dy for earlier layer
            dy = g.pop("dx")
            gradients["layer{}".format(idx)] = g
        return gradients

    def predict_classes(self, x):
        # take max prob classification and reshape to (N, 1)
        set_learning_phase(False)
        out, _ = self.forward_pass(x)
        return softmax(out).argmax(axis=1)

    def eval_metrics(self, y_pred, y_true):
        """
        :param y_pred: predicted labels
        :param y_true: true labels
        :return: values of calculated metrics
        """
        metric_measures = defaultdict(float)
        for metric in self._metrics:
            metric_measures[metric.name] = metric(y_pred, y_true)
        return metric_measures

    def compute_loss(self, out, y):
        return self._loss_func(out, y)

    def load_variables(self, filename):
        with open(filename, "rb") as f:
            vars = pickle.load(f)
        for idx, layer in enumerate(self._layers):
            layer.load_variables(vars["layer{} {}".format(idx, layer.__str__())])

    def save_variables(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.get_variables(), f)

    def get_variables(self):
        vars = {}
        for idx, layer in enumerate(self._layers):
            vars["layer{} {}".format(idx, layer.__str__())] = layer.get_variables()
        return vars

    def param_count(self):
        count = 0
        for l in self._layers:
            for var in l.get_variables().values():
                count += np.prod(var.shape)
        return count

    def output_shape(self):
        return self._layers[-1].output_shape()

    def dump(self, filename):
        with open(filename, "w") as f:
            f.write("params count: %d \n" % (self.param_count()))
            f.write("optimizer: {}\n".format(self._optimizer.name))
            f.write("metric: {}\n".format([metric.name for metric in self._metrics]))

            for layer, vars in self.get_variables().items():
                desc = "{}: ".format(layer)
                if not vars:
                    desc += "no variables"
                else:
                    for name, value in vars.items():
                        desc += "{} {}, ".format(name, value.shape)
                f.write("{}\n".format(desc))
