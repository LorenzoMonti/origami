import numpy as np
import pytest

pytest.importorskip("timestral")

from origami.models import get_encoder, registry


@pytest.fixture(scope="module")
def enc():
    return get_encoder("timestral-5m", device="cpu")


def _curves():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 300)
    short = rng.normal(size=48).astype(np.float32)
    long = (np.sin(2 * np.pi * t / 0.7) + 0.1 * rng.normal(size=300)).astype(np.float32)
    odd = rng.normal(size=33).astype(np.float32)  # one point past a patch boundary
    return [short, long, odd]


def _ones(vs):
    return [np.ones(len(v), bool) for v in vs]


def _times(vs):
    return [np.arange(len(v), dtype=np.float64) for v in vs]


def test_registry():
    spec = registry()["timestral-209m"]
    assert spec["extra"] == "timestral" and spec["uses_time"] is False and spec["revision"]


def test_shape_finite_deterministic(enc):
    vs = _curves()
    a = enc.encode(vs, _ones(vs), _times(vs))
    b = enc.encode(vs, _ones(vs), _times(vs))
    assert a.shape == (3, enc.model.config.d_model) and np.isfinite(a).all()
    assert np.array_equal(a, b)
    assert not np.allclose(a[0], a[1])


def test_batch_independent(enc):
    vs = _curves()
    batched = enc.encode(vs, _ones(vs), _times(vs), batch_size=3)
    for i, v in enumerate(vs):
        alone = enc.encode([v], _ones([v]), _times([v]))
        assert np.allclose(alone[0], batched[i], atol=1e-4)


def test_mask_restricts_pooling(enc):
    v = _curves()[1]
    m = np.zeros(len(v), bool)
    m[-64:] = True  # only the last two patches (300 -> 320 after left padding) are pooled
    full = enc.encode([v], [np.ones(len(v), bool)], [np.arange(300.0)])
    part = enc.encode([v], [m], [np.arange(300.0)])
    assert np.isfinite(part).all() and not np.allclose(full, part)
