import numpy as np

from utils import to_one_hot


def mean_squared_error(scores, y):
    y_onehot = to_one_hot(y)
    error = y_onehot - scores
    loss = np.mean(error ** 2)
    dscores = error / len(y)
    return loss, dscores


def categorical_cross_entropy(scores, y):
    """
    Implementation based on: http://cs231n.github.io/neural-networks-case-study/
    :param scores: Model output scores of shape (N, M), where M is the channel count.
    :param y: Correct classes of shape (N)
    :return: tuple with loss value and loss function gradients w.r.t. scores.
    """
    probs = softmax(scores)
    correct_logscores = -np.log(probs[range(len(y)), y])
    loss = np.mean(correct_logscores)
    dscores = probs
    dscores[range(len(y)), y] -= 1
    dscores = -dscores / len(y)
    return loss, dscores


def softmax(x):
    """
    :param x: Input array of shape (N, M) where M is the channel count.
    """
    x = x - np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=1, keepdims=True)
