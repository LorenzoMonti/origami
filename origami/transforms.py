"""
Input representations: how a light curve is turned into the 1D sequence fed to
a time-series model.

Each transform returns Sequences(values, masks, times, scales): lists of
float32 arrays (possibly of different lengths), boolean arrays marking the
positions that come from real data, the time of each position in days (used by
models that take timestamps), and per curve the standard deviation in
magnitudes removed by z-normalisation (values * scale restores magnitude
units, for models trained on unscaled magnitudes such as Astromer). Embedders
pool only over masked positions.

  raw              magnitudes in time order, z-normalised; times from the first epoch
  folded           magnitudes sorted by phase, z-normalised; times = phase * period
  folded-grid      phase-folded curve, smoothing spline evaluated on a regular
                   grid of `grid` points over one cycle, z-normalised; by default
                   rolled so that maximum light (minimum magnitude) is at phase 0,
                   which makes the sequence independent of the reference epoch;
                   times = grid phase * period
  padded-starembed StarEmbed protocol (Rizhko et al. 2025): last `ctx` points,
                   z-normalised, zero left-padding, every position pooled
"""
from typing import List, NamedTuple

import numpy as np
from scipy.interpolate import make_smoothing_spline

GRID = 200
CTX_STAREMBED = 200


def znorm(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


OVERSMOOTH = 0.5  # spline residual std / data std above which the GCV fit is considered collapsed
MAX_HARMONICS = 10


def spline_grid(phase, mag, grid=GRID):
    """Smoothing spline (GCV) of mag(phase) on a regular grid over [0, 1).

    The GCV choice of the smoothing parameter can collapse to an almost linear fit,
    increasingly often with many points per curve (~1% at 60 points, ~25% at 200
    on synthetic sinusoids). Such fits, and failed ones, are replaced by a Fourier
    series with the number of harmonics chosen by BIC.
    """
    order = np.argsort(phase)
    x, y = phase[order], mag[order]
    x, uniq = np.unique(x, return_index=True)
    y = y[uniq]
    g = np.linspace(0, 1, grid, endpoint=False)
    try:
        f = make_smoothing_spline(x, y)
        if np.std(y - f(x)) <= OVERSMOOTH * np.std(y):
            return f(g)
    except Exception:
        pass
    return fourier_grid(x, y, g)


def fourier_grid(x, y, g, max_harmonics=MAX_HARMONICS):
    """Least-squares Fourier series of y(x), number of harmonics by BIC, evaluated at g."""
    def design(ph, k):
        cols = [np.ones_like(ph)]
        for j in range(1, k + 1):
            cols += [np.cos(2 * np.pi * j * ph), np.sin(2 * np.pi * j * ph)]
        return np.column_stack(cols)

    n, best = len(x), None
    for k in range(1, min(max_harmonics, (n - 2) // 2) + 1):
        coef, *_ = np.linalg.lstsq(design(x, k), y, rcond=None)
        rss = np.sum((y - design(x, k) @ coef) ** 2)
        bic = n * np.log(rss / n + 1e-300) + (2 * k + 1) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, k, coef)
    if best is None:
        return np.interp(g, x, y, period=1.0)
    return design(g, best[1]) @ best[2]


class Sequences(NamedTuple):
    values: List[np.ndarray]
    masks: List[np.ndarray]
    times: List[np.ndarray]
    scales: np.ndarray


def _std(x):
    s = np.std(x)
    return s if s > 0 else 1.0


def raw(lcs):
    seqs = [znorm(c.mag) for c in lcs]
    return _out(seqs, [np.ones(len(s), bool) for s in seqs], [c.time - c.time[0] for c in lcs],
                [_std(c.mag) for c in lcs])


def folded(lcs):
    seqs, times = [], []
    for c in lcs:
        ph = c.phase()
        order = np.argsort(ph)
        seqs.append(znorm(c.mag[order]))
        times.append(ph[order] * c.period)
    return _out(seqs, [np.ones(len(s), bool) for s in seqs], times, [_std(c.mag) for c in lcs])


def folded_grid(lcs, grid=GRID, align="max-light"):
    """align: "max-light" (roll the grid so the brightest point comes first) or None (keep the epoch's phase)."""
    if align not in ("max-light", None):
        raise ValueError(f"align must be 'max-light' or None, got {align!r}")
    g = np.linspace(0, 1, grid, endpoint=False)
    seqs, times, scales = [], [], []
    for c in lcs:
        s = spline_grid(c.phase(), c.mag, grid)
        if align == "max-light":
            s = np.roll(s, -int(np.argmin(s)))
        seqs.append(znorm(s))
        times.append(g * c.period)
        scales.append(_std(s))
    return _out(seqs, [np.ones(grid, bool) for _ in seqs], times, scales)


def padded_starembed(lcs, ctx=CTX_STAREMBED):
    seqs, masks, times = [], [], []
    for c in lcs:
        z = znorm(c.mag)[-ctx:]
        t = (c.time - c.time[0])[-ctx:]
        seqs.append(np.concatenate([np.zeros(ctx - len(z)), z]))
        times.append(np.concatenate([np.zeros(ctx - len(t)), t]))
        masks.append(np.ones(ctx, bool))
    return _out(seqs, masks, times, [_std(c.mag) for c in lcs])


def _out(seqs, masks, times, scales):
    return Sequences([s.astype(np.float32) for s in seqs], masks, [np.asarray(t, np.float64) for t in times],
                     np.asarray(scales, np.float64))


TRANSFORMS = {
    "raw": raw,
    "folded": folded,
    "folded-grid": folded_grid,
    "padded-starembed": padded_starembed,
}


def apply(name, lcs, **kwargs):
    if name not in TRANSFORMS:
        raise ValueError(f"unknown input transform {name!r}; available: {sorted(TRANSFORMS)}")
    return TRANSFORMS[name](lcs, **kwargs)
