import numpy as np
import pytest

pytest.importorskip("chronos")

from origami.models import get_encoder


@pytest.mark.parametrize("name", ["chronos-t5-tiny", "chronos-2-small"])
def test_batch_independent_and_finite(name):
    enc = get_encoder(name, device="cpu")
    rng = np.random.default_rng(0)
    short = rng.normal(size=48).astype(np.float32)
    long = rng.normal(size=300).astype(np.float32)
    ones = lambda v: np.ones(len(v), bool)
    alone = enc.encode([short], [ones(short)], [np.arange(48.0)])
    batched = enc.encode([short, long], [ones(short), ones(long)], [np.arange(48.0), np.arange(300.0)])
    assert alone.shape[0] == 1 and batched.shape[0] == 2 and np.isfinite(batched).all()
    assert np.allclose(alone[0], batched[0], atol=1e-4)
