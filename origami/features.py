"""
Non-learned representations: hand-crafted light-curve features, Fourier
decomposition on the folded curve, and columns taken from a catalogue.

Every representation exposes `name` and `transform(lcs) -> (n_curves, n_features)`.
"""
import numpy as np


class HandCrafted:
    """Feature sets computed from the light curve itself.

    "light-curve": time-domain statistics from the `light-curve` package
                   (period-free; needs pip install origami-astro[features])
    "fourier":     amplitudes A1..Ak and cos/sin of phi21, phi31 from a
                   k-harmonic fit to the folded curve (needs periods)
    "physical":    log10 period and peak-to-peak amplitude of the folded curve:
                   what the z-normalised, phase-folded input of an embedder
                   discards (apparent magnitude is left out on purpose: it
                   depends on distance and survey selection, not on the star). Combine it with an embedding
                   (compare(..., combine=[("chronos", "physical")])) so that
                   embeddings and features see the same information
    """

    KINDS = ("light-curve", "fourier", "physical")

    def __init__(self, kind="light-curve", n_harmonics=3):
        if kind not in self.KINDS:
            raise ValueError(f"unknown feature set {kind!r}; available: {self.KINDS}")
        self.kind, self.n_harmonics = kind, n_harmonics

    @property
    def name(self):
        return f"fourier[k={self.n_harmonics}]" if self.kind == "fourier" else self.kind

    def transform(self, lcs):
        if self.kind == "light-curve":
            return light_curve_features(lcs)
        if self.kind == "fourier":
            return fourier_features(lcs, self.n_harmonics)
        return physical_features(lcs)


class Catalog:
    """Columns of an external table, row-aligned with the light curves (e.g. period, phi31 from a release)."""

    def __init__(self, table, columns, name=None):
        self.values = np.asarray(table[columns] if hasattr(table, "loc") else table, dtype=np.float64)
        self.values = self.values.reshape(len(self.values), -1)
        self.columns = list(columns)
        self._name = name

    @property
    def name(self):
        return self._name or "catalog[" + ",".join(map(str, self.columns)) + "]"

    def transform(self, lcs):
        if len(lcs) != len(self.values):
            raise ValueError(f"{self.name}: {len(self.values)} rows for {len(lcs)} light curves")
        return self.values


def fourier_features(lcs, n_harmonics=3):
    """Least-squares Fourier series m = a0 + sum_k a_k cos(2 pi k phase) + b_k sin(2 pi k phase)."""
    if n_harmonics < 3:
        raise ValueError("phi31 needs at least 3 harmonics")
    rows = []
    for c in lcs:
        ph = c.phase()
        cols = [np.ones_like(ph)]
        for k in range(1, n_harmonics + 1):
            cols += [np.cos(2 * np.pi * k * ph), np.sin(2 * np.pi * k * ph)]
        coef, *_ = np.linalg.lstsq(np.column_stack(cols), c.mag, rcond=None)
        a, b = coef[1::2], coef[2::2]
        phase = np.arctan2(-b, a)
        phi21, phi31 = phase[1] - 2 * phase[0], phase[2] - 3 * phase[0]
        rows.append([*np.hypot(a, b), np.cos(phi21), np.sin(phi21), np.cos(phi31), np.sin(phi31)])
    return np.array(rows)


def physical_features(lcs):
    """log10(period) and peak-to-peak amplitude of the folded-curve fit."""
    from .transforms import spline_grid
    return np.array([[np.log10(c.period), np.ptp(spline_grid(c.phase(), c.mag))] for c in lcs])


def light_curve_features(lcs):
    try:
        import light_curve as lc
    except ImportError as e:
        raise ImportError("hand-crafted features need: pip install origami-astro[features]") from e
    ext = lc.Extractor(
        lc.Amplitude(), lc.AndersonDarlingNormal(), lc.BeyondNStd(nstd=1), lc.BeyondNStd(nstd=2),
        lc.Cusum(), lc.Eta(), lc.EtaE(), lc.InterPercentileRange(0.25), lc.InterPercentileRange(0.1),
        lc.Kurtosis(), lc.LinearFit(), lc.LinearTrend(), lc.MagnitudePercentageRatio(),
        lc.MaximumSlope(), lc.Mean(), lc.Median(), lc.MedianAbsoluteDeviation(),
        lc.MedianBufferRangePercentage(), lc.OtsuSplit(), lc.PercentAmplitude(), lc.ReducedChi2(),
        lc.Skew(), lc.StandardDeviation(), lc.StetsonK(), lc.WeightedMean(),
    )
    feats = [ext(c.time, c.mag, c.err, sorted=True, check=False) for c in lcs]
    return np.nan_to_num(np.array(feats), nan=0.0, posinf=0.0, neginf=0.0)
