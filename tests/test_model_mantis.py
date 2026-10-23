import numpy as np
import pytest

pytest.importorskip("mantis")

from origami.models import get_encoder, registry

NAMES = [n for n, s in registry().items() if s["module"] == "mantis"]


@pytest.fixture(scope="module", params=NAMES)
def enc(request):
    return request.param, get_encoder(request.param, device="cpu")


def _curves(lengths, seed=0):
    rng = np.random.default_rng(seed)
    vals = [rng.normal(size=n).astype(np.float32) for n in lengths]
    return vals, [np.ones(n, bool) for n in lengths], [np.arange(n, dtype=np.float64) for n in lengths]


def test_shape_finite_deterministic(enc):
    name, e = enc
    v, m, t = _curves([48, 200, 700])
    a, b = e.encode(v, m, t), e.encode(v, m, t)
    dim = 512 if registry()[name]["output_token"] == "combined" else 256
    assert a.shape == (3, dim) and np.isfinite(a).all()
    assert np.array_equal(a, b)


def test_batch_independent(enc):
    _, e = enc
    v, m, t = _curves([48, 300, 1000, 5])
    alone = e.encode(v[:1], m[:1], t[:1])
    batched = e.encode(v, m, t, batch_size=3)
    assert np.allclose(alone[0], batched[0], atol=1e-4)


def test_masked_positions_ignored(enc):
    _, e = enc
    v, m, t = _curves([60])
    padded = np.concatenate([np.zeros(40, np.float32), v[0]])
    mask = np.concatenate([np.zeros(40, bool), m[0]])
    a = e.encode(v, m, t)
    b = e.encode([padded], [mask], [np.arange(100.0)])
    assert np.allclose(a, b, atol=1e-5)


def test_truncates_to_most_recent(enc):
    _, e = enc
    v, m, t = _curves([900])
    a = e.encode(v, m, t)
    b = e.encode([v[0][-512:]], [m[0][-512:]], [t[0][-512:]])
    assert np.allclose(a, b, atol=1e-5)
