import numpy as np
import pytest

pytest.importorskip("toto2")

from origami.models import get_encoder, registry


@pytest.fixture(scope="module")
def enc():
    return get_encoder("toto-2.0-4m", device="cpu")


def _curves():
    rng = np.random.default_rng(0)
    short = rng.normal(size=48).astype(np.float32)   # 2 patches, 16 padding points
    long = rng.normal(size=300).astype(np.float32)   # 10 patches
    gappy = rng.normal(size=200).astype(np.float32)
    m_gappy = np.ones(200, bool)
    m_gappy[50:120] = False
    gappy[~m_gappy] = np.nan
    return [short, long, gappy], [np.ones(48, bool), np.ones(300, bool), m_gappy]


def test_shape_finite_deterministic(enc):
    v, m = _curves()
    t = [np.arange(len(x), dtype=float) for x in v]
    a = enc.encode(v, m, t)
    b = enc.encode(v, m, t)
    assert a.shape == (3, registry()["toto-2.0-4m"]["dim"])
    assert np.isfinite(a).all()
    assert np.array_equal(a, b)


def test_batch_independent(enc):
    v, m = _curves()
    t = [np.arange(len(x), dtype=float) for x in v]
    batched = enc.encode(v, m, t)
    for i in range(3):
        alone = enc.encode([v[i]], [m[i]], [t[i]])
        assert np.allclose(alone[0], batched[i], atol=1e-4)
    # batch_size splitting gives the same result
    assert np.allclose(enc.encode(v, m, t, batch_size=1), batched, atol=1e-4)


def test_truncation_keeps_last_points(enc):
    rng = np.random.default_rng(1)
    ctx = registry()["toto-2.0-4m"]["context"]
    x = rng.normal(size=ctx + 100).astype(np.float32)
    full = enc.encode([x], [np.ones(len(x), bool)], [np.arange(len(x), dtype=float)])
    tail = enc.encode([x[-ctx:]], [np.ones(ctx, bool)], [np.arange(ctx, dtype=float)])
    assert np.allclose(full, tail, atol=1e-5)
