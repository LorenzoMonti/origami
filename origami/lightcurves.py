"""
Container for sets of irregularly sampled, single-band light curves.

Every curve keeps its own time, magnitude and error arrays (ragged), an
identifier, a provenance tag (survey and band, e.g. "ztf"/"r"), and
optionally a period, a reference epoch and the origin of the period
("catalog" or "estimated"). Epoch subsampling returns a new set, so the
same subsampled curves can be passed to every representation being compared.
"""
import hashlib
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class LightCurve:
    time: np.ndarray
    mag: np.ndarray
    err: np.ndarray
    id: object = None
    survey: str = None
    band: str = None
    period: float = None
    epoch: float = None
    period_source: str = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        t, m, e = (np.asarray(a, dtype=np.float64) for a in (self.time, self.mag, self.err))
        if not (t.shape == m.shape == e.shape) or t.ndim != 1:
            raise ValueError(f"light curve {self.id}: time, mag and err must be 1D arrays of equal length")
        order = np.argsort(t, kind="stable")
        object.__setattr__(self, "time", t[order])
        object.__setattr__(self, "mag", m[order])
        object.__setattr__(self, "err", e[order])

    def __len__(self):
        return len(self.time)

    def phase(self):
        """Phase in [0, 1) from the period and the reference epoch (first observation if no epoch)."""
        if self.period is None:
            raise ValueError(f"light curve {self.id} has no period: set one with LightCurves.with_periods")
        t0 = self.time[0] if self.epoch is None else self.epoch
        return ((self.time - t0) / self.period) % 1.0


class LightCurves:
    """An ordered set of LightCurve objects."""

    def __init__(self, curves):
        self.curves = list(curves)

    @classmethod
    def from_arrays(cls, times, mags, errs, ids=None, survey=None, band=None, periods=None, epochs=None,
                    period_source="catalog"):
        """survey and band: a single value for the whole set or one per curve."""
        n = len(times)
        ids = range(n) if ids is None else ids
        surveys, bands = _per_curve(survey, n), _per_curve(band, n)
        periods = [None] * n if periods is None else periods
        epochs = [None] * n if epochs is None else epochs
        return cls(LightCurve(t, m, e, id=i, survey=sv, band=b, period=p, epoch=ep,
                              period_source=None if p is None else period_source)
                   for t, m, e, i, sv, b, p, ep in zip(times, mags, errs, ids, surveys, bands, periods, epochs))

    @classmethod
    def from_frame(cls, df, id_col="id", time_col="time", mag_col="mag", err_col="err", ids=None,
                   survey=None, band=None):
        """Long-format table, one row per observation. `ids` fixes the order (default: order of appearance)."""
        groups = dict(tuple(df.groupby(id_col, sort=False)))
        ids = list(groups) if ids is None else list(ids)
        return cls(LightCurve(groups[i][time_col].values, groups[i][mag_col].values,
                              groups[i][err_col].values, id=i, survey=survey, band=band) for i in ids)

    @classmethod
    def from_csv_dir(cls, directory, ids=None, time_col="time", mag_col="mag", err_col="err",
                     flux_col=None, flux_err_col=None, survey=None, band=None, pattern="{id}.csv", **read_kwargs):
        """One CSV file per object. `ids` fixes the set and order (default: every file matching the
        pattern, sorted). With flux_col/flux_err_col, magnitude errors are 1.0857 * flux_err / flux and
        mag_col is still used for the magnitudes."""
        import glob
        import os

        import pandas as pd
        if ids is None:
            prefix, suffix = pattern.split("{id}")
            files = sorted(glob.glob(os.path.join(directory, prefix + "*" + suffix)))
            ids = [os.path.basename(f)[len(prefix):len(os.path.basename(f)) - len(suffix)] for f in files]
        curves = []
        for i in ids:
            d = pd.read_csv(os.path.join(directory, pattern.format(id=i)), **read_kwargs)
            err = (1.0857 * d[flux_err_col].values / d[flux_col].values) if flux_col else d[err_col].values
            curves.append(LightCurve(d[time_col].values, d[mag_col].values, err, id=i, survey=survey, band=band))
        return cls(curves)

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.curves[idx]
        if isinstance(idx, slice):
            return LightCurves(self.curves[idx])
        return LightCurves(self.curves[i] for i in idx)

    def __iter__(self):
        return iter(self.curves)

    @property
    def ids(self):
        return np.array([c.id for c in self.curves])

    @property
    def periods(self):
        return np.array([np.nan if c.period is None else c.period for c in self.curves])

    @property
    def provenance(self):
        """Distinct (survey, band) pairs in the set."""
        return sorted({(c.survey, c.band) for c in self.curves}, key=str)

    def n_epochs(self):
        return np.array([len(c) for c in self.curves])

    def with_periods(self, periods, epochs=None, source="catalog"):
        epochs = [c.epoch for c in self.curves] if epochs is None else epochs
        return LightCurves(replace(c, period=float(p), epoch=ep, period_source=source)
                           for c, p, ep in zip(self.curves, periods, epochs))

    def subsample(self, k, seed=0):
        """Keep k random epochs per curve (all of them if a curve has fewer than k).

        The selection for curve i depends only on (seed, i), so it is reproducible
        and independent of the other curves in the set.
        """
        if k is None or k == "all":
            return self
        out = []
        for i, c in enumerate(self.curves):
            if len(c) <= k:
                out.append(c)
                continue
            rng = np.random.default_rng([seed, i])
            idx = np.sort(rng.choice(len(c), size=k, replace=False))
            out.append(replace(c, time=c.time[idx], mag=c.mag[idx], err=c.err[idx]))
        return LightCurves(out)

    def fingerprint(self):
        """Hash of the data, periods and epochs (used as cache key for embeddings)."""
        h = hashlib.sha256()
        for c in self.curves:
            h.update(f"{c.id}|{c.survey}|{c.band}".encode())
            for a in (c.time, c.mag, c.err):
                h.update(np.ascontiguousarray(a).tobytes())
            h.update(np.float64(np.nan if c.period is None else c.period).tobytes())
            h.update(np.float64(np.nan if c.epoch is None else c.epoch).tobytes())
        return h.hexdigest()[:16]


def _per_curve(value, n):
    if value is None or isinstance(value, str):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"expected one value per curve ({n}), got {len(value)}")
    return list(value)
