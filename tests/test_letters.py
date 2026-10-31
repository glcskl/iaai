"""Тесты классификатора букв: softmax, форма весов, предсказания."""
import numpy as np

import letters_net


def test_softmax_normalises_rows():
    probs = letters_net.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), rtol=1e-12)
    assert np.all(probs >= 0)


def test_softmax_is_stable_for_large_logits():
    probs = letters_net.softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.all(np.isfinite(probs))
    assert probs[0, 0] > 0.999


def test_saved_weights_have_expected_shapes():
    weights = np.load(letters_net.MODEL_PATH)
    assert weights["W1"].shape == (784, letters_net.HIDDEN)
    assert weights["b1"].shape == (1, letters_net.HIDDEN)
    assert weights["W2"].shape == (letters_net.HIDDEN, letters_net.N_CLASSES)
    assert weights["b2"].shape == (1, letters_net.N_CLASSES)


def test_predict_returns_valid_classes_and_probabilities():
    images = np.random.default_rng(0).random((4, 784))
    classes, probs = letters_net.predict(images)
    assert classes.shape == (4,)
    assert np.all((classes >= 0) & (classes < letters_net.N_CLASSES))
    assert probs.shape == (4, letters_net.N_CLASSES)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), rtol=1e-5)


def test_accuracy_is_one_for_its_own_predictions():
    weights = np.load(letters_net.MODEL_PATH)
    W1, b1, W2, b2 = (weights[key] for key in ("W1", "b1", "W2", "b2"))
    images = np.random.default_rng(1).random((8, 784))
    logits = np.maximum(images @ W1 + b1, 0) @ W2 + b2
    labels = logits.argmax(axis=1)
    assert letters_net.accuracy(W1, b1, W2, b2, images, labels) == 1.0
