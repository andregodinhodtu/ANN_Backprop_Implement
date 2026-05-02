import sys
sys.path.append("src/base_python_version")
import random
import statistics
import pytest
from ANN_base_python import ANN_base_python as ANN


def make_ann(seed=0):
    """Build a small reproducible 4 → 5 → 2 network."""
    rng = random.Random(seed)
    return ANN(
        n_layers=3,
        n_neurons_each_layer=[4, 5, 2],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
        rng=rng,
    )


def make_random_batch(seed, n_samples, n_features=4, n_outputs=2):
    """Build (X, Y) as lists of column vectors for the base-Python compute_loss."""
    rng = random.Random(seed)
    X = [
        [[rng.gauss(0, 1)] for _ in range(n_features)]
        for _ in range(n_samples)
    ]
    Y = [
        [[float(rng.randint(0, 1))] for _ in range(n_outputs)]
        for _ in range(n_samples)
    ]
    return X, Y


# ============================================================
# compute_loss — happy path
# ============================================================
def test_compute_loss_returns_float():
    ann = make_ann()
    X = [[[1.0]] * 4 for _ in range(3)]   # 3 samples, each (4, 1)
    Y = [[[0.0]] * 2 for _ in range(3)]   # 3 targets, each (2, 1)

    loss = ann.compute_loss(X, Y)
    assert isinstance(loss, float)


def test_compute_loss_is_non_negative():
    """BCE is always >= 0."""
    ann = make_ann()
    X, Y = make_random_batch(seed=0, n_samples=5)
    assert ann.compute_loss(X, Y) >= 0.0


def test_compute_loss_single_vs_batch_consistency():
    """Mean loss over a batch == mean of per-sample losses."""
    ann = make_ann(seed=42)
    X, Y = make_random_batch(seed=1, n_samples=6)

    batched = ann.compute_loss(X, Y)
    per_sample = [ann.compute_loss([X[i]], [Y[i]]) for i in range(len(X))]

    assert batched == pytest.approx(statistics.mean(per_sample))
