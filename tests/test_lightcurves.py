import numpy as np
import pytest

from origami import LightCurves, transforms


def make_set(n=5, n_obs=60, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    times, mags, errs, periods = [], [], [], []
    for _ in range(n):
        p = rng.uniform(0.3, 0.8)
        t = np.sort(rng.uniform(0, 1000, n_obs))
        m = 15 + 0.4 * np.cos(2 * np.pi * t / p) + rng.normal(0, noise, n_obs)
        times.append(t), mags.append(m), errs.append(np.full(n_obs, noise)), periods.append(p)
    return LightCurves.from_arrays(times, mags, errs, survey="synthetic", band="G", periods=periods)


def test_sorted_by_time():
    lcs = LightCurves.from_arrays([np.array([3.0, 1.0, 2.0])], [np.array([30.0, 10.0, 20.0])], [np.ones(3)])
    assert np.all(np.diff(lcs[0].time) > 0)
    assert np.array_equal(lcs[0].mag, [10.0, 20.0, 30.0])


def test_subsample_reproducible_and_independent_of_set():
    lcs = make_set()
    a, b = lcs.subsample(20, seed=1), lcs.subsample(20, seed=1)
    assert all(np.array_equal(x.time, y.time) for x, y in zip(a, b))
    assert np.all(a.n_epochs() == 20)
    # the selection for a curve does not depend on the other curves in the set
    assert np.array_equal(a[0].time, lcs[[0]].subsample(20, seed=1)[0].time)
    assert not np.array_equal(a[0].time, lcs.subsample(20, seed=2)[0].time)


def test_subsample_keeps_short_curves():
    lcs = make_set(n_obs=10)
    assert np.all(lcs.subsample(20).n_epochs() == 10)


def test_fingerprint_changes_with_data():
    lcs = make_set()
    assert lcs.fingerprint() == make_set().fingerprint()
    assert lcs.fingerprint() != lcs.subsample(30).fingerprint()


def test_phase_requires_period():
    lcs = LightCurves.from_arrays([np.arange(5.0)], [np.zeros(5)], [np.ones(5)])
    with pytest.raises(ValueError):
        lcs[0].phase()


@pytest.mark.parametrize("name", sorted(transforms.TRANSFORMS))
def test_transforms_shapes(name):
    lcs = make_set()
    seqs, masks, times, scales = transforms.apply(name, lcs)
    assert len(seqs) == len(masks) == len(times) == len(scales) == len(lcs) and np.all(scales > 0)
    for s, m, t in zip(seqs, masks, times):
        assert s.dtype == np.float32 and len(s) == len(m) == len(t) and np.isfinite(s).all()


def test_folded_grid_recovers_shape():
    lcs = make_set(n=1, n_obs=200)
    seq, *_ = transforms.apply("folded-grid", lcs, align=None)
    g = np.linspace(0, 1, transforms.GRID, endpoint=False)
    t0 = lcs[0].time[0]
    expected = transforms.znorm(np.cos(2 * np.pi * (g + t0 / lcs[0].period)))
    assert np.corrcoef(seq[0], expected)[0, 1] > 0.99


def test_folded_grid_alignment_removes_epoch_dependence():
    lcs = make_set(n=1, n_obs=200)
    shifted = lcs.with_periods(lcs.periods, epochs=[lcs[0].time[0] + 0.37 * lcs[0].period])
    a, *_ = transforms.apply("folded-grid", lcs)
    b, *_ = transforms.apply("folded-grid", shifted)
    assert np.argmin(a[0]) == 0 and np.argmin(b[0]) == 0
    assert np.corrcoef(a[0], b[0])[0, 1] > 0.999


def test_provenance():
    lcs = make_set(n=2)
    assert lcs.provenance == [("synthetic", "G")]
    mixed = LightCurves.from_arrays([np.arange(3.0)] * 2, [np.zeros(3)] * 2, [np.ones(3)] * 2,
                                    survey=["gaia_dr3", "ztf"], band=["G", "r"])
    assert mixed.provenance == [("gaia_dr3", "G"), ("ztf", "r")]
    assert mixed.subsample(2)[1].survey == "ztf"
    relabelled = LightCurves.from_arrays([np.arange(3.0)] * 2, [np.zeros(3)] * 2, [np.ones(3)] * 2,
                                         survey="ztf", band=["G", "r"])
    assert relabelled.fingerprint() != mixed.fingerprint()


def test_folded_grid_never_flattens_dense_curves():
    # with ~200 points the GCV spline alone collapses on about a quarter of these curves
    for seed in range(20):
        lcs = make_set(n=1, n_obs=200, seed=seed)
        for shift in (0.0, 0.37):
            moved = lcs.with_periods(lcs.periods, epochs=[lcs[0].time[0] + shift * lcs[0].period])
            seq, *_ = transforms.apply("folded-grid", moved)
            g = np.linspace(0, 1, transforms.GRID, endpoint=False)
            assert np.corrcoef(seq[0], -np.cos(2 * np.pi * g))[0, 1] > 0.98  # magnitude: brightest = minimum


def test_from_csv_dir(tmp_path):
    import pandas as pd
    lcs = make_set(n=3)
    for c in lcs:
        pd.DataFrame(dict(time=c.time[::-1], mag=c.mag[::-1], flux=10 ** (-0.4 * c.mag[::-1]),
                          flux_error=0.01 * 10 ** (-0.4 * c.mag[::-1])))\
            .to_csv(tmp_path / f"{c.id}.csv", index=False)
    back = LightCurves.from_csv_dir(tmp_path, ids=lcs.ids, flux_col="flux", flux_err_col="flux_error",
                                    survey="gaia_dr3", band="G")
    assert np.allclose(back[1].mag, lcs[1].mag) and np.allclose(back[1].err, 0.010857)
    assert back.provenance == [("gaia_dr3", "G")]
    assert len(LightCurves.from_csv_dir(tmp_path, flux_col="flux", flux_err_col="flux_error")) == 3
