import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("huggingface_hub")

from origami.models import get_encoder, registry
from origami.models.astromer import Encoder

NAMES = ["astromer1-macho", "astromer2-macho", "astromer1-ztfdr20"]


def curve(rng, n, period=0.55):
    t = np.sort(rng.uniform(0, 1000, n))
    v = np.cos(2 * np.pi * t / period) + 0.3 * np.cos(4 * np.pi * t / period + 0.5) + rng.normal(0, 0.05, n)
    return ((v - v.mean()) / v.std()).astype(np.float32), np.ones(n, bool), t


@pytest.fixture(scope="module", params=NAMES)
def enc(request):
    return get_encoder(request.param, device="cpu")


def test_registry():
    for name in NAMES:
        spec = registry()[name]
        assert spec["uses_time"] and spec["context"] == 200 and spec["extra"] == "astromer"


def test_shape_finite_deterministic_batch_independent(enc):
    rng = np.random.default_rng(0)
    curves = [curve(rng, n) for n in (30, 120, 450)]  # 450 > window: last 200 used
    v, m, t = map(list, zip(*curves))
    full = enc.encode(v, m, t)
    assert full.shape == (3, 256) and full.dtype == np.float32 and np.isfinite(full).all()
    assert np.array_equal(full, enc.encode(v, m, t))
    for i in range(3):
        alone = enc.encode([v[i]], [m[i]], [t[i]])
        assert np.allclose(alone[0], full[i], atol=1e-5)
    assert np.allclose(enc.encode(v, m, t, batch_size=1), full, atol=1e-5)


def test_times_matter(enc):
    rng = np.random.default_rng(1)
    v, m, t = curve(rng, 80)
    a = enc.encode([v], [m], [t])
    b = enc.encode([v], [m], [t * 3.0])
    assert not np.allclose(a, b, atol=1e-4)
    # only time differences matter (times are centred per window)
    assert np.allclose(a, enc.encode([v], [m], [t + 12345.0]), atol=1e-4)


def test_masked_positions_ignored_and_last_window(enc):
    rng = np.random.default_rng(2)
    v, m, t = curve(rng, 60)
    m2 = m.copy()
    m2[::3] = False
    junk = v.copy()
    junk[::3] = 99.0
    assert np.allclose(enc.encode([junk], [m2], [t]), enc.encode([v[m2]], [m[m2]], [t[m2]]), atol=1e-5)
    v, m, t = curve(rng, 350)
    assert np.allclose(enc.encode([v], [m], [t]), enc.encode([v[-200:]], [m[-200:]], [t[-200:]]), atol=1e-5)


def test_window_centring():
    v = np.array([1.0, 2.0, 3.0], np.float32)
    inp, tim, msk = Encoder.window(v, np.ones(3, bool), np.array([10.0, 11.0, 15.0]))
    assert inp.shape == (200,) and msk.sum() == 3
    assert np.allclose(inp[:3], [-1, 0, 1]) and np.allclose(tim[:3], [-2, -1, 3]) and not inp[3:].any()
