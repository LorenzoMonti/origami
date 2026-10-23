"""
Downstream tasks used to probe a representation. Each task takes the feature
matrix X, the targets y and a fold index, and returns (metrics, output):

  regression      standardized RidgeCV, out-of-fold predictions; RMSE, MAE
  classification  standardized logistic regression (balanced class weights),
                  out-of-fold predictions; balanced accuracy, macro F1
  clustering      no training labels: standardized (optionally PCA-reduced)
                  features clustered with k-means, HDBSCAN or CLiMB; ARI and
                  NMI against y (folds are not used)
  anomaly         y = 1 for anomalies, 0 for normal objects; a detector is fitted
                  on the normal objects of the training folds and scores the
                  held-out fold; AUROC, average precision

Folds: integers; fold f is predicted by a model trained on every other
non-negative fold. Rows with a negative fold are left out entirely, and
`eval_folds` restricts which folds are predicted and scored. An explicit
train/test split is folds = 0 (train) / 1 (test) / -1 (unused) with
eval_folds = [1]. Metrics are computed on the predicted rows only.

Distance-based tasks (clustering, anomaly) receive `blocks`, the column count of
each concatenated representation: after standardization every block is divided
by sqrt(its width), so that each block has the same total variance. Without it a
256-dimensional embedding outweighs a 2-column feature block by 128 to 1.
"""
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (adjusted_rand_score, average_precision_score, balanced_accuracy_score, f1_score,
                             normalized_mutual_info_score, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-3, 4, 15)
PRIMARY_METRIC = {"regression": "rmse", "classification": "balanced_accuracy", "clustering": "ari",
                  "anomaly": "auroc"}


def split_folds(folds, eval_folds=None):
    """(evaluated folds, mask of predicted rows); negative folds are never used."""
    folds = np.asarray(folds)
    if eval_folds is None:
        eval_folds = [f for f in np.unique(folds) if f >= 0]
    return list(eval_folds), np.isin(folds, eval_folds)


def _train_test(folds, f):
    return (folds >= 0) & (folds != f), folds == f


def regression(X, y, folds, clip=True, eval_folds=None):
    """clip: predictions limited to the training label range (guards against
    degenerate features on poorly sampled curves)."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.full(len(y), np.nan)
    evals, done = split_folds(folds, eval_folds)
    for f in evals:
        tr, te = _train_test(folds, f)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)).fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        if clip:
            pred[te] = np.clip(pred[te], y[tr].min(), y[tr].max())
    per_fold = [_rmse(y[folds == f], pred[folds == f]) for f in evals]
    return dict(score("regression", y, pred, done), rmse_fold_sd=float(np.std(per_fold))), pred


def classification(X, y, folds, C=1.0, eval_folds=None):
    y = np.asarray(y)
    pred = np.empty(len(y), dtype=object)
    evals, done = split_folds(folds, eval_folds)
    for f in evals:
        tr, te = _train_test(folds, f)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
        pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
    per_fold = [balanced_accuracy_score(y[folds == f], pred[folds == f].astype(y.dtype)) for f in evals]
    return dict(score("classification", y, pred, done),
                balanced_accuracy_fold_sd=float(np.std(per_fold))), pred


def balance_blocks(Z, blocks):
    """Divide each standardized block of columns by sqrt(its width)."""
    if blocks is None or len(blocks) < 2:
        return Z
    weights = np.concatenate([np.full(b, 1 / np.sqrt(b)) for b in blocks])
    return Z * weights


def clustering(X, y, folds=None, method="kmeans", n_clusters=None, pca=None, random_state=0, blocks=None,
               eval_folds=None, **kwargs):
    """method: "kmeans" (n_clusters defaults to the number of classes in y), "hdbscan", or "climb"
    (needs the climb-astro package; kwargs go to CLiMB). Noise points (-1) count as one extra cluster."""
    Z = balance_blocks(StandardScaler().fit_transform(X), blocks)
    if pca:
        Z = PCA(n_components=min(pca, Z.shape[1]), random_state=random_state).fit_transform(Z)
    k = n_clusters or len(np.unique(y))
    if method == "kmeans":
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(Z)
    elif method == "hdbscan":
        labels = HDBSCAN(**{"min_cluster_size": max(5, len(Z) // 100), **kwargs}).fit_predict(Z)
    elif method == "climb":
        try:
            from CLiMB.core.CLiMB import CLiMB
        except ImportError as e:
            raise ImportError("method='climb' needs: pip install climb-astro") from e
        labels = np.asarray(CLiMB(**{"constrained_clusters": k, **kwargs}).fit(Z).get_labels())
    else:
        raise ValueError(f"unknown clustering method {method!r}; available: 'kmeans', 'hdbscan', 'climb'")
    return dict(score("clustering", y, labels, np.ones(len(y), bool)), n_clusters=int(len(set(labels) - {-1})),
                noise_frac=float(np.mean(labels == -1))), labels


def score(task, y, out, mask):
    """Task metrics on the rows selected by mask (also used for per-group metrics)."""
    y, out = np.asarray(y)[mask], np.asarray(out)[mask]
    n = int(mask.sum())
    if task == "regression":
        return dict(rmse=_rmse(y, out), mae=float(np.mean(np.abs(y - out))), n=n)
    if task == "classification":
        out = out.astype(y.dtype)
        return dict(balanced_accuracy=float(balanced_accuracy_score(y, out)),
                    macro_f1=float(f1_score(y, out, average="macro")), n=n)
    if task == "clustering":
        return dict(ari=float(adjusted_rand_score(y, out)), nmi=float(normalized_mutual_info_score(y, out)), n=n)
    if len(np.unique(y)) < 2:  # anomaly metrics need both classes in the subset
        return dict(auroc=np.nan, average_precision=np.nan, anomaly_frac=float(np.mean(y)), n=n)
    return dict(auroc=float(roc_auc_score(y, out)), average_precision=float(average_precision_score(y, out)),
                anomaly_frac=float(np.mean(y)), n=n)


def anomaly(X, y, folds, detector="knn", n_neighbors=10, random_state=0, blocks=None, eval_folds=None):
    """detector: "knn" (mean distance to the k nearest normal training objects) or "isolation-forest"."""
    y = np.asarray(y).astype(int)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("anomaly task: y must be 1 for anomalies and 0 for normal objects")
    out = np.full(len(y), np.nan)
    evals, done = split_folds(folds, eval_folds)
    for f in evals:
        tr, te = _train_test(folds, f)
        tr &= y == 0
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = balance_blocks(scaler.transform(X[tr]), blocks), balance_blocks(scaler.transform(X[te]), blocks)
        if detector == "knn":
            dist, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(Xtr).kneighbors(Xte)
            out[te] = dist.mean(1)
        elif detector == "isolation-forest":
            out[te] = -IsolationForest(random_state=random_state).fit(Xtr).score_samples(Xte)
        else:
            raise ValueError(f"unknown detector {detector!r}; available: 'knn', 'isolation-forest'")
    return score("anomaly", y, out, done), out


DISTANCE_BASED = ("clustering", "anomaly")

TASKS = {"regression": regression, "classification": classification, "clustering": clustering, "anomaly": anomaly}


def _rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))
