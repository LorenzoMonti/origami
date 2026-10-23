import numpy as np
import pytest

pytest.importorskip("uni2ts")
torch = pytest.importorskip("torch")

from origami.models import get_encoder, registry  # noqa: E402

NAMES = ["moirai-1.1-small", "moirai-2.0-small"]


def _curves(lengths, seed=0):
    rng = np.random.default_rng(seed)
    values, masks, times = [], [], []
    for n in lengths:
        t = np.sort(rng.uniform(0, 1000, n))
        x = np.sin(2 * np.pi * t / rng.uniform(0.3, 1.0)) + 0.1 * rng.normal(size=n)
        values.append(((x - x.mean()) / x.std()).astype(np.float32))
        masks.append(np.ones(n, bool))
        times.append(t)
    return values, masks, times


@pytest.fixture(scope="module", params=NAMES)
def encoder(request):
    try:
        return get_encoder(request.param, device="cpu")
    except Exception as e:  # weights not downloadable (offline)
        pytest.skip(f"cannot load {request.param}: {e}")


def test_registered():
    reg = registry()
    for n in NAMES:
        assert reg[n]["extra"] == "moirai" and reg[n]["uses_time"] is False


def test_shape_finite_deterministic(encoder):
    v, m, t = _curves([37, 200, 5, 64])
    e1 = encoder.encode(v, m, t)
    e2 = encoder.encode(v, m, t)
    assert e1.shape == (4, encoder.module.d_model)
    assert np.isfinite(e1).all()
    np.testing.assert_array_equal(e1, e2)


def test_padding_invariance(encoder):
    v, m, t = _curves([50, 300, 17, 129], seed=1)
    alone = np.concatenate([encoder.encode([v[i]], [m[i]], [t[i]]) for i in range(len(v))])
    batched = encoder.encode(v, m, t)
    np.testing.assert_allclose(alone, batched, atol=1e-4, rtol=1e-4)
    # sub-batching must not matter either
    np.testing.assert_allclose(encoder.encode(v, m, t, batch_size=2), batched, atol=1e-4, rtol=1e-4)


def test_masked_points_change_embedding(encoder):
    v, m, t = _curves([96], seed=2)
    m2 = [m[0].copy()]
    m2[0][:40] = False
    a = encoder.encode(v, m, t)
    b = encoder.encode(v, m2, t)
    assert not np.allclose(a, b)
    # a fully masked leading patch is excluded from pooling; values under the mask are ignored
    v3 = [v[0].copy()]
    v3[0][:40] = 123.0
    np.testing.assert_allclose(encoder.encode(v3, m2, t), b, atol=1e-5)


def test_truncation_keeps_most_recent(encoder):
    ctx = encoder.spec["context"]
    v, m, t = _curves([ctx + 100], seed=3)
    full = encoder.encode(v, m, t, batch_size=1)
    tail = encoder.encode([v[0][-ctx:]], [m[0][-ctx:]], [t[0][-ctx:]], batch_size=1)
    np.testing.assert_allclose(full, tail, atol=1e-5)


def test_matches_module_forward(encoder):
    """The hidden states we pool are those the full model feeds to its output head."""
    v, m, _ = _curves([48], seed=4)
    p = encoder.spec["patch"]
    tgt = torch.from_numpy(v[0]).view(1, -1, p)
    obs = torch.ones_like(tgt, dtype=torch.bool)
    T = tgt.shape[1]
    sid, tid = torch.ones((1, T), dtype=torch.long), torch.arange(T).view(1, T)
    captured = {}
    h = encoder.module.encoder.register_forward_hook(lambda mod, inp, out: captured.setdefault("h", out))
    try:
        args = (tgt, obs, sid, tid, torch.zeros_like(sid), torch.zeros((1, T), dtype=torch.bool))
        with torch.no_grad():
            if encoder.spec["family"] == "moirai-2":
                encoder.module(*args, training_mode=False)
            else:  # the full model takes patches zero-padded to its largest patch size
                pad = max(encoder.module.patch_sizes) - p
                padded = (torch.nn.functional.pad(tgt, (0, pad)), torch.nn.functional.pad(obs, (0, pad)))
                encoder.module(*padded, *args[2:], torch.full_like(tid, p))
    finally:
        h.remove()
    with torch.no_grad():
        ours = encoder._hidden(tgt, obs, sid, tid)
    torch.testing.assert_close(ours, captured["h"])
