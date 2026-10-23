"""
Compare representations with the same task, the same folds and the same
(sub)sampled light curves.

    res = compare(lcs, y, {"chronos": Embedder("chronos-t5-tiny"), "fourier": HandCrafted("fourier")},
                  task="regression", combine=[("chronos", "fourier")], subsample_epochs=["all", 40, 20])
    res.table()
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .tasks import DISTANCE_BASED, PRIMARY_METRIC, TASKS, score


class Results:
    def __init__(self, records, outputs, task, provenance):
        self.records = pd.DataFrame(records)
        self.outputs = outputs  # {(representation, epochs, seed): predictions / cluster labels / anomaly scores}
        self.task = task
        self.provenance = provenance

    @property
    def predictions(self):
        return self.outputs

    def table(self, metric=None):
        """Whole evaluated set, mean over subsampling seeds; columns are epoch levels."""
        metric = metric or PRIMARY_METRIC[self.task]
        rec = self.records[self.records.group == "all"]
        return rec.groupby(["representation", "epochs"], sort=False)[metric].mean().unstack("epochs")

    def group_table(self, metric=None, epochs="all"):
        """Per-group metric (compare(..., groups=...)) at one epoch level; columns are groups."""
        metric = metric or PRIMARY_METRIC[self.task]
        rec = self.records[(self.records.group != "all") & (self.records.epochs == epochs)]
        return rec.groupby(["representation", "group"], sort=False)[metric].mean().unstack("group")

    def __repr__(self):
        return (f"Results(task={self.task!r}, metric={PRIMARY_METRIC[self.task]!r}, provenance={self.provenance})\n"
                f"{self.table().round(4).to_string()}")


def make_folds(y, task, n_folds=5, random_state=42):
    folds = np.zeros(len(y), int)
    stratified = task in ("classification", "anomaly")
    splitter = (StratifiedKFold if stratified else KFold)(n_folds, shuffle=True, random_state=random_state)
    for k, (_, te) in enumerate(splitter.split(np.zeros(len(y)), y if stratified else None)):
        folds[te] = k
    return folds


def split_to_folds(train, test):
    """Boolean train/test masks -> folds (0 train, 1 test, -1 unused) and eval_folds=[1]."""
    train, test = np.asarray(train, bool), np.asarray(test, bool)
    if np.any(train & test):
        raise ValueError(f"split: {int(np.sum(train & test))} objects are in both train and test")
    folds = np.full(len(train), -1)
    folds[train], folds[test] = 0, 1
    return folds, [1]


def compare(lcs, y, representations, task="regression", combine=(), folds=None, n_folds=5,
            subsample_epochs=("all",), seeds=(0,), random_state=42, split=None, groups=None, **task_kwargs):
    """
    lcs               LightCurves
    y                 targets row-aligned with lcs: values (regression), class labels
                      (classification, clustering), 1 = anomaly / 0 = normal (anomaly)
    representations   {name: object with transform(lcs)}
    task              "regression", "classification", "clustering" or "anomaly"
    combine           tuples of names whose features are concatenated, e.g. [("chronos", "fourier")]
    folds             fold index per curve (e.g. folds of a published model); default
                      shuffled KFold, stratified for classification and anomaly
    subsample_epochs  epoch levels; "all" keeps every observation
    seeds             subsampling seeds per level (ignored for "all")
    split             (train_mask, test_mask): train on one subset, score the other (e.g. bright vs
                      faint, one survey vs another) instead of cross-validation; supervised and anomaly tasks
    groups            one label per curve (e.g. magnitude bin, survey); metrics are also reported per group
                      (Results.group_table)
    task_kwargs       passed to the task, e.g. method="hdbscan" or detector="isolation-forest"
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; available: {sorted(TASKS)}")
    y = np.asarray(y)
    if len(y) != len(lcs):
        raise ValueError(f"{len(y)} targets for {len(lcs)} light curves")
    eval_folds = None
    if split is not None:
        if folds is not None:
            raise ValueError("give either folds or split, not both")
        if task == "clustering":
            raise ValueError("split does not apply to clustering (no training step)")
        folds, eval_folds = split_to_folds(*split)
    folds = make_folds(y, task, n_folds, random_state) if folds is None else np.asarray(folds)
    if groups is not None and len(groups) != len(y):
        raise ValueError(f"{len(groups)} group labels for {len(y)} light curves")
    for group in combine:
        missing = set(group) - set(representations)
        if missing:
            raise ValueError(f"combine {group}: unknown representations {sorted(missing)}")

    records, outputs = [], {}
    for level in subsample_epochs:
        for seed in ([0] if level == "all" else seeds):
            sub = lcs.subsample(level, seed=seed)
            feats = {name: rep.transform(sub) for name, rep in representations.items()}
            todo = [(name, feats[name], None) for name in representations]
            todo += [(" + ".join(g), np.hstack([feats[n] for n in g]), [feats[n].shape[1] for n in g])
                     for g in combine]
            for name, X, blocks in todo:
                kwargs = dict(task_kwargs, blocks=blocks) if task in DISTANCE_BASED else dict(task_kwargs)
                if eval_folds is not None:
                    kwargs["eval_folds"] = eval_folds
                metrics, out = TASKS[task](X, y, folds, **kwargs)
                base = dict(representation=name, epochs=level, seed=seed, dim=X.shape[1])
                records.append(dict(base, group="all", **metrics))
                outputs[(name, level, seed)] = out
                if groups is not None:
                    done = np.ones(len(y), bool) if task == "clustering" else np.isin(folds, eval_folds or
                                                                                      [f for f in np.unique(folds) if f >= 0])
                    for g in np.unique(np.asarray(groups)[done]):
                        records.append(dict(base, group=g, **score(task, y, out, done & (np.asarray(groups) == g))))
    return Results(records, outputs, task, lcs.provenance)
