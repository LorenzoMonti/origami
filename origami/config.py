"""
Experiment configuration (YAML) for the command-line workflow.

    name: my_experiment
    output: /path/to/runs/my_experiment          # embeddings, results, logs
    data:
      light_curves: /path/to/light_curves   # one CSV per object, file name "{id}.csv"
      catalog: /path/to/catalog.csv         # one row per object
      id_col: source_id                     # in the catalogue; ids also name the CSV files
      time_col: time
      mag_col: mag
      err_col: mag_err                      # or flux_col + flux_err_col
      flux_col: null
      flux_err_col: null
      survey: ztf                           # provenance tag
      band: r
      period_col: period                    # optional (needed by folded inputs and Fourier features)
      epoch_col: epoch                      # optional
      fold_col: fold                        # optional, otherwise shuffled (stratified) 5-fold
    representations:                        # name: one of {model, features, catalog}
      chronos: {model: chronos-t5-tiny, input: folded-grid}
      moment:                               # `setup`: shell line run by its SLURM task before origami,
        model: moment-1-small               # for models whose dependencies need their own environment
        setup: source /path/venvs/moment/bin/activate
      fourier: {features: fourier}
      physical: {features: physical}
      P: {catalog: [period]}
      my_vae:                               # any user model: an object with transform(lcs) -> (n, dim)
        python: my_package.encoders:MyVAE   # "module:attribute", called with kwargs
        path: /home/me/vae_project          # optional, added to sys.path
        kwargs: {checkpoint: /path/vae.pt}
        version: v3                         # optional; change it to invalidate stored embeddings
    combine: [[chronos, fourier, P], [chronos, physical]]
    tasks:                                  # target: catalogue column; other keys go to the task
      - {task: regression, target: my_target}
      - {task: clustering, target: class, method: kmeans}
      - task: classification
        target: class
        split: {train: "mag < 19", test: "mag >= 19"}   # pandas queries on the catalogue (or column/train/test)
        groups: {column: mag, bins: [16, 18, 19, 20, 21]}  # per-group metrics (or just a column name)
    subsample_epochs: [all, 40, 30, 20]
    seeds: [0, 1, 2]
"""
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from .compare import compare
from .embedders import Embedder, MissingEmbedding
from .features import Catalog, HandCrafted
from .lightcurves import LightCurves
from .tasks import TASKS

DATA_KEYS = {"light_curves", "catalog", "id_col", "time_col", "mag_col", "err_col", "flux_col", "flux_err_col",
             "survey", "band", "period_col", "epoch_col", "fold_col", "pattern"}


@dataclass
class Experiment:
    name: str
    output: str
    data: dict
    representations: dict
    tasks: list
    combine: list = field(default_factory=list)
    subsample_epochs: list = field(default_factory=lambda: ["all"])
    seeds: list = field(default_factory=lambda: [0])

    @property
    def embeddings_dir(self):
        return os.path.join(self.output, "embeddings")

    @property
    def results_dir(self):
        return os.path.join(self.output, "results")

    def model_representations(self):
        """Representations computed by embed jobs (foundation models and user Python models)."""
        return [n for n, r in self.representations.items() if "model" in r or "python" in r]

    def models(self):
        return sorted({r["model"] for r in self.representations.values() if "model" in r})

    def load(self):
        """LightCurves (with periods/epochs if configured) and the row-aligned catalogue."""
        import pandas as pd
        d = self.data
        cat = pd.read_csv(d["catalog"])
        ids = cat[d.get("id_col", "id")].values
        lcs = LightCurves.from_csv_dir(d["light_curves"], ids=ids, time_col=d.get("time_col", "time"),
                                       mag_col=d.get("mag_col", "mag"), err_col=d.get("err_col", "err"),
                                       flux_col=d.get("flux_col"), flux_err_col=d.get("flux_err_col"),
                                       survey=d.get("survey"), band=d.get("band"),
                                       pattern=d.get("pattern", "{id}.csv"))
        if d.get("period_col"):
            epochs = cat[d["epoch_col"]].values if d.get("epoch_col") else None
            lcs = lcs.with_periods(cat[d["period_col"]].values, epochs=epochs)
        return lcs, cat

    def representation(self, name, cat, compute=True, device=None):
        r = dict(self.representations[name])
        r.pop("setup", None)  # used only by the SLURM script
        if "model" in r:
            model, inp = r.pop("model"), r.pop("input", "folded-grid")
            batch = r.pop("batch_size", 256)
            return Embedder(model, input=inp, device=device, batch_size=batch, cache_dir=self.embeddings_dir,
                            compute=compute, **r)
        if "python" in r:
            return PythonRepresentation(name, r, cache_dir=self.embeddings_dir, compute=compute)
        if "features" in r:
            return HandCrafted(r["features"], **{k: v for k, v in r.items() if k != "features"})
        return Catalog(cat, r["catalog"], name=name)

    def run_tasks(self, compute=False, device=None, log=print):
        """Run every configured task; embeddings are read from disk unless compute=True."""
        lcs, cat = self.load()
        reps = {n: self.representation(n, cat, compute=compute, device=device) for n in self.representations}
        folds = cat[self.data["fold_col"]].values if self.data.get("fold_col") else None
        results = []
        for t in self.tasks:
            t = dict(t)
            options = {k: v for k, v in t.items() if k not in ("task", "target")}
            task, target = t.pop("task"), t.pop("target")
            split = _split(cat, t.pop("split")) if "split" in t else None
            groups = _groups(cat, t.pop("groups")) if "groups" in t else None
            y = cat[target].values
            log(f"[{self.name}] {task} on {target!r} ({len(y)} objects)")
            res = compare(lcs, y, reps, task=task, combine=[tuple(g) for g in self.combine],
                          folds=None if split is not None else folds, split=split, groups=groups,
                          subsample_epochs=self.subsample_epochs, seeds=self.seeds, **t)
            results.append((task, target, options, res))
        return results


class PythonRepresentation:
    """A user model given as "module:attribute" in the config, with on-disk caching like Embedder."""

    def __init__(self, name, spec, cache_dir, compute=True):
        self._name, self.spec, self.cache_dir, self.compute = name, spec, cache_dir, compute
        self._obj = None

    @property
    def name(self):
        return self._name

    def _cache_path(self, lcs):
        key = json.dumps([self.spec.get("python"), self.spec.get("kwargs", {}), self.spec.get("version"),
                          lcs.fingerprint()], sort_keys=True, default=str)
        digest = hashlib.sha256(key.encode()).hexdigest()[:20]
        return os.path.join(self.cache_dir, "python", self._name, f"{digest}.npy")

    def _load(self):
        if self._obj is None:
            if self.spec.get("path"):
                sys.path.insert(0, self.spec["path"])
            module, _, attr = self.spec["python"].partition(":")
            factory = getattr(importlib.import_module(module), attr)
            self._obj = factory(**self.spec.get("kwargs", {}))
            if not hasattr(self._obj, "transform"):
                raise TypeError(f"{self.spec['python']} must build an object with transform(lcs)")
        return self._obj

    def transform(self, lcs):
        path = self._cache_path(lcs)
        if os.path.exists(path):
            return np.load(path)
        if not self.compute:
            raise MissingEmbedding(f"{self._name}: no stored embedding for this data ({path})")
        emb = np.asarray(self._load().transform(lcs), dtype=np.float64)
        if emb.shape[0] != len(lcs):
            raise ValueError(f"{self._name}: transform returned {emb.shape[0]} rows for {len(lcs)} light curves")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, emb)
        return emb


def _split(cat, spec):
    """{column, train, test} (values of a column) or {train, test} (pandas queries) -> boolean masks."""
    if "column" in spec:
        col = cat[spec["column"]]
        return col.isin(np.atleast_1d(spec["train"])).values, col.isin(np.atleast_1d(spec["test"])).values
    idx = cat.index
    return idx.isin(cat.query(spec["train"]).index), idx.isin(cat.query(spec["test"]).index)


def _groups(cat, spec):
    """A column name, or {column, bins} to bin a numeric column."""
    import pandas as pd
    if isinstance(spec, str):
        return cat[spec].astype(str).values
    return pd.cut(cat[spec["column"]], bins=spec["bins"]).astype(str).values


def load_config(path):
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    missing = {"name", "output", "data", "representations", "tasks"} - set(raw)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    unknown = set(raw["data"]) - DATA_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown data keys {sorted(unknown)}; allowed: {sorted(DATA_KEYS)}")
    for key in ("light_curves", "catalog"):
        if key not in raw["data"]:
            raise ValueError(f"{path}: data.{key} is required")
    for name, r in raw["representations"].items():
        kinds = {"model", "features", "catalog", "python"} & set(r)
        if len(kinds) != 1:
            raise ValueError(f"{path}: representation {name!r} needs exactly one of model / features / catalog / python")
    for group in raw.get("combine", []):
        bad = set(group) - set(raw["representations"])
        if bad:
            raise ValueError(f"{path}: combine {group} uses unknown representations {sorted(bad)}")
    for t in raw["tasks"]:
        if t.get("task") not in TASKS or "target" not in t:
            raise ValueError(f"{path}: each task needs task (one of {sorted(TASKS)}) and target, got {t}")
    epochs = [e if e == "all" else int(e) for e in raw.get("subsample_epochs", ["all"])]
    return Experiment(name=raw["name"], output=raw["output"], data=raw["data"],
                      representations=raw["representations"], tasks=raw["tasks"],
                      combine=raw.get("combine", []), subsample_epochs=epochs, seeds=raw.get("seeds", [0]))


def embedding_jobs(exp):
    """(representation, epochs, seed) combinations an embed job has to compute."""
    return [(n, level, seed) for n in exp.model_representations() for level in exp.subsample_epochs
            for seed in ([0] if level == "all" else exp.seeds)]


def epochs_label(level):
    return "all" if level == "all" else str(int(level))


__all__ = ["Experiment", "load_config", "embedding_jobs", "epochs_label", "np"]
