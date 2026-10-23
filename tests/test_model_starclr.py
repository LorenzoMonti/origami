import numpy as np
import pytest

from origami.models import registry
from origami.models.starclr import preprocess


def curve(rng, n, period=0.55):
    t = np.sort(rng.uniform(0, 1000, n))
    v = np.cos(2 * np.pi * t / period) + 0.3 * np.cos(4 * np.pi * t / period + 0.5) + rng.normal(0, 0.05, n)
    return ((v - v.mean()) / v.std()).astype(np.float32), np.ones(n, bool), t


def test_registry():
    spec = registry()["starclr"]
    assert spec["uses_time"] and spec["extra"] == "starclr" and spec["context"] == 8192
    assert spec["checksum"].startswith("sha256:") and "zenodo" in spec["source"]


def test_preprocess_matches_official_protocol():
    rng = np.random.default_rng(0)
    v, m, t = curve(rng, 300)
    v[10] = 8.0  # faint outlier in magnitude: removed by the 3-sigma clip
    m[::7] = False
    tt, f = preprocess(v, m, t + 100.0)
    flux = -v[m].astype(np.float64)
    keep = np.abs(flux - flux.mean()) <= 3 * flux.std()
    assert not keep[np.flatnonzero(m).tolist().index(10)]
    lo, hi = np.percentile(flux[keep], [5, 95])
    assert np.allclose(f, (flux[keep] - lo) / (hi - lo), atol=1e-6)
    assert tt[0] == 0 and np.allclose(tt, t[m][keep] - t[m][keep][0], atol=1e-3)
    # brighter (lower magnitude) -> larger input
    assert np.corrcoef(f, -v[m][keep])[0, 1] > 0.999
    # truncation keeps the most recent real points
    tt, f = preprocess(v, np.ones(300, bool), t, context=50, clip_sigma=np.inf)
    assert len(f) == 50 and np.allclose(tt, t[-50:] - t[-50], atol=1e-3)
    assert preprocess(v, np.zeros(300, bool), t)[1].size == 0


pytest.importorskip("torch")
pytest.importorskip("safetensors")


@pytest.fixture(scope="module")
def enc():
    from origami.models import get_encoder
    return get_encoder("starclr", device="cpu")


def test_shape_finite_deterministic_batch_independent(enc):
    rng = np.random.default_rng(0)
    curves = [curve(rng, n) for n in (30, 120, 450)]
    v, m, t = map(list, zip(*curves))
    full = enc.encode(v, m, t)
    assert full.shape == (3, 512) and full.dtype == np.float32 and np.isfinite(full).all()
    assert np.array_equal(full, enc.encode(v, m, t))
    for i in range(3):
        assert np.allclose(enc.encode([v[i]], [m[i]], [t[i]])[0], full[i], atol=1e-4)
    assert np.allclose(enc.encode(v, m, t, batch_size=1), full, atol=1e-4)
    assert np.allclose(enc.encode(v[::-1], m[::-1], t[::-1])[::-1], full, atol=1e-4)


def test_times_matter(enc):
    rng = np.random.default_rng(1)
    v, m, t = curve(rng, 80)
    a = enc.encode([v], [m], [t])
    assert not np.allclose(a, enc.encode([v], [m], [t * 3.0]), atol=1e-3)
    assert np.allclose(a, enc.encode([v], [m], [t + 500.0]), atol=1e-4)  # times start from their minimum


def test_masks_empty_and_truncation(enc):
    rng = np.random.default_rng(2)
    v, m, t = curve(rng, 60)
    m2 = m.copy()
    m2[::3] = False
    junk = v.copy()
    junk[::3] = 99.0
    assert np.allclose(enc.encode([junk], [m2], [t]), enc.encode([v[m2]], [m[m2]], [t[m2]]), atol=1e-4)
    e = enc.encode([v, v[:0]], [m, m[:0]], [t, t[:0]])
    assert np.isfinite(e).all() and not e[1].any()
    enc.spec = dict(enc.spec, context=100)
    try:
        v, m, t = curve(rng, 250)
        assert np.allclose(enc.encode([v], [m], [t]), enc.encode([v[-100:]], [m[-100:]], [t[-100:]]), atol=1e-4)
    finally:
        enc.spec = dict(enc.spec, context=8192)
