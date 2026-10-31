"""Тесты сети из network.py: формы, диапазоны, градиенты и обучение на XOR."""
import numpy as np

from network import NeuralNet

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([[0], [1], [1], [0]], dtype=float)


def numerical_gradients(net, X, y, eps=1e-5):
    """Центральные разности по каждому параметру — эталон для backprop."""
    grads = {}
    for name in ("W1", "b1", "W2", "b2"):
        param = getattr(net, name)
        grad = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            plus = net.loss(net.forward(X), y)
            param[idx] = original - eps
            minus = net.loss(net.forward(X), y)
            param[idx] = original
            grad[idx] = (plus - minus) / (2 * eps)
        grads[name] = grad
    return grads


def test_parameters_have_expected_shapes():
    net = NeuralNet(3, 4, 1, seed=7)
    assert net.W1.shape == (3, 4)
    assert net.b1.shape == (1, 4)
    assert net.W2.shape == (4, 1)
    assert net.b2.shape == (1, 1)


def test_forward_returns_probabilities():
    net = NeuralNet(3, 4, 1, seed=7)
    out = net.forward(np.random.default_rng(0).standard_normal((16, 3)))
    assert out.shape == (16, 1)
    assert np.all(out > 0) and np.all(out < 1)


def test_predict_returns_booleans_of_same_shape():
    net = NeuralNet(2, 4, 1, seed=7)
    pred = net.predict(XOR_X)
    assert pred.shape == (4, 1)
    assert pred.dtype == np.bool_


def test_loss_is_near_zero_for_confident_correct_predictions():
    net = NeuralNet(2, 4, 1, seed=7)
    y = np.array([[1.0], [0.0]])
    assert net.loss(np.array([[0.999999], [0.000001]]), y) < 1e-5


def test_loss_is_large_for_confident_wrong_predictions():
    net = NeuralNet(2, 4, 1, seed=7)
    y = np.array([[1.0], [0.0]])
    assert net.loss(np.array([[0.000001], [0.999999]]), y) > 5


def test_backward_matches_numerical_gradients():
    rng = np.random.default_rng(3)
    net = NeuralNet(3, 5, 1, seed=11)
    X = rng.standard_normal((8, 3))
    y = (rng.random((8, 1)) > 0.5).astype(float)

    net.forward(X)
    analytic = dict(zip(("W1", "b1", "W2", "b2"), net.backward(X, y)))
    numeric = numerical_gradients(net, X, y)
    for name, grad in analytic.items():
        np.testing.assert_allclose(grad, numeric[name], rtol=1e-4, atol=1e-6)


def test_training_solves_xor():
    net = NeuralNet(2, 4, 1, seed=42)
    losses = net.train(XOR_X, XOR_Y, epochs=3000, lr=1.0, verbose=False)
    assert len(losses) == 3000
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.1
    assert net.predict(XOR_X).tolist() == [[False], [True], [True], [False]]
