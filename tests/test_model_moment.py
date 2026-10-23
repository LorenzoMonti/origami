import numpy as np
import pytest

pytest.importorskip("momentfm")
pytest.importorskip("torch")

from origami import models  # noqa: E402

NAME = "moment-1-small"


@pytest.fixture(scope="module")
def encoder():
    return models.get_encoder(NAME, device="cpu")


def curves(lengths, seed=0):
    rng = np.random.default_rng(seed)
    values, masks, times = [], [], []
    for n in lengths:
        t = np.sort(rng.uniform(0, 100, n))
        v = np.sin(2 * np.pi * t / rng.uniform(0.3, 2.0)) + rng.normal(0, 0.1, n)
        values.append(((v - v.mean()) / v.std()).astype(np.float32))
        masks.append(np.ones(n, bool))
        times.append(t)
    return values, masks, times


def test_registry_spec():
    spec = models.registry()[NAME]
    assert spec["context"] == 512 and spec["patch"] == 8 and spec["uses_time"] is False
    assert spec["license"] == "MIT"


def test_shape_finite_deterministic(encoder):
    v, m, t = curves([5, 37, 200, 700])  # shorter than a patch, unaligned, aligned, longer than context
    e1 = encoder.encode(v, m, t, batch_size=2)
    e2 = encoder.encode(v, m, t, batch_size=2)
    assert e1.shape == (4, 512)
    assert np.all(np.isfinite(e1))
    np.testing.assert_array_equal(e1, e2)


def test_batching_with_longer_curves_does_not_change_embedding(encoder):
    v, m, t = curves([60, 300, 512, 1000], seed=1)
    alone = encoder.encode(v[:1], m[:1], t[:1])
    batched = encoder.encode(v, m, t)
    np.testing.assert_allclose(batched[0], alone[0], rtol=1e-4, atol=1e-5)


def test_masked_points_are_ignored(encoder):
    v, m, t = curves([160], seed=2)
    # Append 40 masked points (5 whole patches) of garbage at the end: same embedding.
    v2 = [np.concatenate([np.zeros(40, np.float32) + 99, v[0]])]
    m2 = [np.concatenate([np.zeros(40, bool), m[0]])]
    t2 = [np.concatenate([np.zeros(40), t[0]])]
    np.testing.assert_allclose(encoder.encode(v2, m2, t2)[0], encoder.encode(v, m, t)[0], rtol=1e-4, atol=1e-5)


def test_truncation_keeps_most_recent_points(encoder):
    v, m, t = curves([800], seed=3)
    tail = ([v[0][-512:]], [m[0][-512:]], [t[0][-512:]])
    np.testing.assert_allclose(encoder.encode(v, m, t)[0], encoder.encode(*tail)[0], rtol=1e-5, atol=1e-6)
