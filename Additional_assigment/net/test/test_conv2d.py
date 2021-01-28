import numpy as np

from net.layers.conv import Conv2D


def test_forward_output_size_3_filters_images_4_4():
    l = Conv2D(filters=2, kernel_size=3)
    l.build((None, 4, 4, 1))
    x = np.zeros((10, 4, 4, 1))
    y = l.forward(x)
    assert y.shape == (10, 2, 2, 2)


def test_forward_output_size_5_filters_3_channels_images_7_7():
    l = Conv2D(filters=5, kernel_size=3)
    l.build((None, 7, 7, 3))
    x = np.zeros((10, 7, 7, 3))
    y = l.forward(x)
    assert y.shape == (10, 5, 5, 5)


def test_forward_output_size_4_filters_2_channels_images_10_6():
    l = Conv2D(filters=4, kernel_size=5)
    l.build((None, 10, 6, 2))
    x = np.zeros((10, 10, 6, 2))
    y = l.forward(x)
    assert y.shape == (10, 6, 2, 4)


def test_forward_results():
    l = Conv2D(filters=2, kernel_size=3)
    l.build((1, 5, 5, 1))
    x = np.moveaxis(np.array(
        [[[[1, 1, 1, 1, 1], [2, 1, 3, 1, 0], [5, 0, 0, 3, 1], [2, 2, 2, 5, 2], [1, 1, 0, 7, 2]]]]),
        1, 3)
    F = np.moveaxis(np.array([[[[2, 1, 3],
                                [0, 1, 2],
                                [1, 1, 1]]],
                              [[[0, 0, 1],
                                [-1, 2, 0],
                                [0, 3, 1]]]]), 1, 3)
    bias = np.zeros((1, 1, 1, 2))
    l.load_variables(dict(F=F, bias=bias))
    y = l.forward(x)
    expected_result = np.moveaxis(np.array(
        [[[[18, 14, 11], [20, 23, 21], [18, 29, 24]], [[1, 9, 10], [6, 12, 23], [5, 12, 32]]]]), 1,
        3)
    assert np.allclose(y, expected_result)
    assert y.shape == (1, 3, 3, 2)


def test_backward_shapes():
    l = Conv2D(kernel_size=3, filters=20)
    l.build((5, 10, 10, 3))
    x = np.zeros((5, 10, 10, 3))
    y = l.forward(x)

    grads = l.backward(x, np.ones_like(y))
    assert grads["dx"].shape == x.shape
    assert grads["dW"].shape == (20, 3, 3, 3)
    assert grads["db"].shape == (1, 1, 1, 20)


def test_backward_shapes_2():
    l = Conv2D(kernel_size=2, filters=2, stride=2)
    l.build((5, 4, 4, 3))
    x = np.zeros((5, 4, 4, 3))
    y = l.forward(x)

    grads = l.backward(x, np.ones_like(y))
    assert grads["dx"].shape == x.shape
    assert grads["dW"].shape == (2, 2, 2, 3)
    assert grads["db"].shape == (1, 1, 1, 2)


def test_backward_pad():
    l = Conv2D(kernel_size=3, filters=2, padding=1)
    l.build((5, 5, 5, 3))
    x = np.zeros((5, 5, 5, 3))
    y = l.forward(x)

    grads = l.backward(x, np.ones_like(y))
    assert grads["dx"].shape == x.shape
    assert grads["dW"].shape == (2, 3, 3, 3)
    assert grads["db"].shape == (1, 1, 1, 2)


def test_backward():
    l = Conv2D(filters=2, kernel_size=3)
    l.build((1, 5, 5, 1))
    x = np.moveaxis(np.array(
        [[[[1, 1, 1, 1, 1],
           [2, 1, 3, 1, 0],
           [5, 0, 0, 3, 1],
           [2, 2, 2, 5, 2],
           [1, 1, 0, 7, 2]]]]), 1, 3)
    F = np.moveaxis(np.array([[[[2, 1, 3],
                                [0, 1, 2],
                                [1, 1, 1]]],
                              [[[0, 0, 1],
                                [-1, 2, 0],
                                [0, 3, 1]]]]), 1, 3)
    bias = np.zeros((1, 1, 1, 2))
    l.load_variables(dict(F=F, bias=bias))
    y = l.forward(x)

    grads = l.backward(x, np.ones_like(y))
    assert grads["dx"].shape == x.shape
    expected_db = np.array([9, 9]).reshape((1, 1, 1, 2))
    expected_dW = np.array([[[14, 11, 11],
                             [17, 17, 17],
                             [13, 20, 22]],
                            [[14, 11, 11],
                             [17, 17, 17],
                             [13, 20, 22]]]).reshape((2, 3, 3, 1))

    assert grads["dW"].shape == (2, 3, 3, 1)
    assert grads["db"].shape == (1, 1, 1, 2)
    assert grads['dx'].shape == (1, 5, 5, 1)
    assert np.allclose(grads["db"], expected_db)
    assert np.allclose(grads["dW"], expected_dW)
