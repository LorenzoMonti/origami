import numpy as np
import pytest

from origami import Catalog, HandCrafted, LightCurves, compare
from origami.features import fourier_features


def make_set(n=200, n_obs=60, seed=0):
    """Synthetic periodic curves whose target is a linear function of period and phi31."""
    rng = np.random.default_rng(seed)
    times, mags, errs, periods, y = [], [], [], [], []
    for _ in range(n):
        p, phi = rng.uniform(0.3, 0.8), rng.uniform(4.0, 5.5)
        t = np.sort(rng.uniform(0, 1000, n_obs))
        x = 2 * np.pi * t / p
        m = 15 + 0.3 * np.cos(x) + 0.1 * np.cos(2 * x + 0.5) + 0.05 * np.cos(3 * x + phi) + rng.normal(0, 0.005, n_obs)
        times.append(t), mags.append(m), errs.append(np.full(n_obs, 0.005)), periods.append(p)
        y.append(-1.0 - 2.0 * p + 0.5 * phi)
    lcs = LightCurves.from_arrays(times, mags, errs, survey="synthetic", band="G", periods=periods, epochs=[0.0] * n)
    return lcs, np.array(y)


def test_fourier_recovers_phi31():
    lcs, _ = make_set(n=5, n_obs=300)
    F = fourier_features(lcs)
    phi31 = np.arctan2(F[:, 6], F[:, 5])  # cos/sin of phi31 -> angle
    a = np.array([[0.3, 0.1, 0.05]])
    assert np.allclose(F[:, :3], a, atol=0.01)
    assert np.all(np.isfinite(phi31))


def test_compare_regression_ranks_informative_features_first():
    lcs, y = make_set()
    period = Catalog(lcs.periods.reshape(-1, 1), ["P"])
    res = compare(lcs, y, {"fourier": HandCrafted("fourier"), "P": period},
                  combine=[("fourier", "P")], subsample_epochs=["all", 30], seeds=(0, 1))
    tab = res.table()
    assert tab.loc["fourier + P", "all"] < tab.loc["P", "all"]
    assert tab.loc["fourier + P", "all"] <= tab.loc["fourier + P", 30]
    assert res.provenance == [("synthetic", "G")]
    assert set(res.records.seed[res.records.epochs == 30]) == {0, 1}


def test_compare_uses_given_folds():
    lcs, y = make_set(n=50)
    folds = np.arange(50) % 3
    res = compare(lcs, y, {"P": Catalog(lcs.periods.reshape(-1, 1), ["P"])}, folds=folds)
    assert len(res.predictions[("P", "all", 0)]) == 50


def test_catalog_row_mismatch():
    lcs, y = make_set(n=10)
    with pytest.raises(ValueError):
        compare(lcs, y, {"P": Catalog(np.zeros((9, 1)), ["P"])})


def test_unknown_combination():
    lcs, y = make_set(n=10)
    with pytest.raises(ValueError):
        compare(lcs, y, {"P": Catalog(lcs.periods.reshape(-1, 1), ["P"])}, combine=[("P", "chronos")])


def make_two_classes(n=150, seed=0):
    """Class 0: sinusoids; class 1: sawtooth-like curves with a different period range."""
    rng = np.random.default_rng(seed)
    times, mags, errs, periods, y = [], [], [], [], []
    for i in range(n):
        c = i % 2
        p = rng.uniform(0.45, 0.7) if c == 0 else rng.uniform(0.25, 0.4)
        t = np.sort(rng.uniform(0, 1000, 60))
        ph = (t / p) % 1
        m = 15 + (0.3 * np.cos(2 * np.pi * ph) if c == 0 else 0.4 * (ph - 0.5)) + rng.normal(0, 0.01, 60)
        times.append(t), mags.append(m), errs.append(np.full(60, 0.01)), periods.append(p), y.append(c)
    return LightCurves.from_arrays(times, mags, errs, periods=periods, epochs=[0.0] * n), np.array(y)


def test_classification_task():
    lcs, y = make_two_classes()
    res = compare(lcs, y, {"fourier": HandCrafted("fourier")}, task="classification")
    assert res.table().loc["fourier", "all"] > 0.9
    assert "macro_f1" in res.records


def test_clustering_task():
    lcs, y = make_two_classes()
    reps = {"P": Catalog(lcs.periods.reshape(-1, 1), ["P"])}
    km = compare(lcs, y, reps, task="clustering")
    assert km.table().loc["P", "all"] > 0.9
    hd = compare(lcs, y, reps, task="clustering", method="hdbscan")
    assert hd.table("nmi").loc["P", "all"] > 0.5


def test_anomaly_task():
    lcs, y = make_two_classes()
    keep = np.r_[np.where(y == 0)[0], np.where(y == 1)[0][:10]]
    res = compare(lcs[keep], y[keep], {"fourier": HandCrafted("fourier")}, task="anomaly")
    assert res.table().loc["fourier", "all"] > 0.9
    iso = compare(lcs[keep], y[keep], {"fourier": HandCrafted("fourier")}, task="anomaly",
                  detector="isolation-forest")
    assert iso.table().loc["fourier", "all"] > 0.8


def test_anomaly_rejects_multiclass():
    lcs, y = make_two_classes(n=20)
    with pytest.raises(ValueError):
        compare(lcs, y + 1, {"P": Catalog(lcs.periods.reshape(-1, 1), ["P"])}, task="anomaly")


def test_physical_features():
    lcs, _ = make_two_classes(n=10)
    X = HandCrafted("physical").transform(lcs)
    assert X.shape == (10, 2)
    assert np.allclose(X[::2, 1], 0.6, atol=0.05)  # class 0: 0.3 mag cosine -> 0.6 peak-to-peak
    assert np.allclose(X[:, 0], np.log10(lcs.periods))


def test_block_balancing_in_distance_tasks():
    """An informative 1-column block must not be drowned by a wide noise block."""
    lcs, y = make_two_classes()
    rng = np.random.default_rng(1)

    class Noise:
        name = "noise"

        def transform(self, lcs):
            return rng.normal(size=(len(lcs), 200))

    reps = {"noise": Noise(), "P": Catalog(lcs.periods.reshape(-1, 1), ["P"])}
    res = compare(lcs, y, reps, task="clustering", combine=[("noise", "P")])
    assert res.table().loc["noise + P", "all"] > 0.5
    assert res.table().loc["noise", "all"] < 0.1


def test_explicit_split_trains_on_train_only():
    lcs, y = make_set(n=120)
    train = np.arange(120) < 80
    reps = {"fourier": HandCrafted("fourier"), "P": Catalog(lcs.periods.reshape(-1, 1), ["P"])}
    res = compare(lcs, y, reps, combine=[("fourier", "P")], split=(train, ~train))
    pred = res.outputs[("fourier + P", "all", 0)]
    assert np.isnan(pred[train]).all() and np.isfinite(pred[~train]).all()
    assert res.records.n.iloc[0] == 40
    with pytest.raises(ValueError):
        compare(lcs, y, reps, split=(train, np.ones(120, bool)))


def test_group_metrics():
    lcs, y = make_two_classes()
    mag_bin = np.where(np.arange(len(y)) < 75, "bright", "faint")
    res = compare(lcs, y, {"fourier": HandCrafted("fourier")}, task="classification", groups=mag_bin)
    gt = res.group_table()
    assert list(gt.columns) == ["bright", "faint"]
    assert res.records.loc[res.records.group == "bright", "n"].iloc[0] == 75
    assert "fourier" in res.table().index
