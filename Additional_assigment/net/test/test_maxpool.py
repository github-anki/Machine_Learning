import numpy as np

from net.layers.pooling import MaxPool2D


def test_forward():
    l = MaxPool2D()
    l.build((1, 4, 4, 1))
    x = np.array([[[[1, 1, 5, 60],
                    [2, 4, 7, 8],
                    [7, 2, 1, 2],
                    [1, 0, 3, 9]]]]).reshape((1, 4, 4, 1))
    y = l.forward(x)
    expected_result = np.array([[[[4, 60],
                                  [7, 9]]]]).reshape((1, 2, 2, 1))
    assert np.allclose(y, expected_result)
    assert y.shape == (1, 2, 2, 1)


def test_output_shape():
    l = MaxPool2D()
    l.build((None, 4, 4, 4))
    assert l.output_shape() == (None, 2, 2, 4)


def test_backward():
    l = MaxPool2D()
    l.build((1, 4, 4, 1))
    dy = np.zeros((1, 2, 2, 1))
    dy.fill(1)
    x = np.array([[[[1, 1, 5, 60],
                    [2, 4, 7, 8],
                    [7, 2, 1, 2],
                    [1, 0, 3, 9]]]]).reshape((1, 4, 4, 1))
    y = l.forward(x)

    expected_dx = np.array([[[[0, 0, 0, 1],
                              [0, 1, 0, 0],
                              [1, 0, 0, 0],
                              [0, 0, 0, 1]]]]).reshape((1, 4, 4, 1))
    grads = l.backward(x, dy)
    assert np.allclose(grads["dx"], expected_dx)
    assert grads["dx"].shape == (1, 4, 4, 1)
