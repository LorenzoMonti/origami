<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/LorenzoMonti/origami/main/docs/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/LorenzoMonti/origami/main/docs/logo.svg" alt="origami" width="300">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/origami-astro/"><img src="https://img.shields.io/pypi/v/origami-astro" alt="PyPI version"></a>
  <a href="https://pypi.org/project/origami-astro/"><img src="https://img.shields.io/pypi/pyversions/origami-astro" alt="Python versions"></a>
  <a href="https://github.com/LorenzoMonti/origami/actions/workflows/ci.yml"><img src="https://github.com/LorenzoMonti/origami/actions/workflows/ci.yml/badge.svg" alt="tests"></a>
  <a href="https://github.com/LorenzoMonti/origami/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/origami-astro" alt="License: MIT"></a>
</p>

<p align="center"><b>Fold, embed, probe.</b><br>
Compare foundation-model embeddings, hand-crafted features and catalogue columns
of astronomical light curves, under identical conditions.</p>

---

`origami` answers one kind of question: **which representation of a light curve carries
the information you need, and how robust is it?** It puts every candidate on the same
footing:
- the same objects;
- the same (sub)sampled epochs;
- the same folds;
- the same simple probe.

Candidates can be embeddings from frozen time-series foundation models, classical features,
values from a catalogue, or your own model. Typical questions:
- Do embeddings from a pretrained model add anything to Fourier parameters for predicting a
  physical quantity?
- Which model separates my classes best, and does that survive when curves have 20
  epochs instead of 200?
- Does a representation learned on bright objects transfer to faint ones, or from one
  survey to another?

`origami` is a benchmarking and selection tool. It does not train foundation models, and
it is not a classifier for producing catalogues.

```bash
pip install "origami-astro[features,chronos]"
```

**Highlights**
- **Light curves:** irregular, single-band, any length, each with a provenance tag
  (survey, band).
- **Models:** 31 checkpoints from 10 model families behind one interface (Chronos,
  Chronos-Bolt, Chronos-2, Moirai, MOMENT, Mantis, Toto 2.0, Timestral, Astromer, StarCLR).
  Weights are downloaded on first use.
- **Tasks:** regression, classification, clustering and anomaly detection. Evaluation can use
  cross-validation, an explicit train/test split, or metrics per group.
- **Robustness tests:** controlled epoch subsampling, identical for every representation.
- **Clusters:** a YAML + command-line workflow runs one SLURM array task per model, works on
  compute nodes without internet, and collects everything into tables at the end.

## Contents
- [Quick start](#quick-start)
- [Concepts](#concepts)
  - [Light curves](#light-curves)
  - [Input transforms](#input-transforms)
  - [Representations](#representations)
  - [Tasks and metrics](#tasks-and-metrics)
- [Command-line workflow](#command-line-workflow)
- [Models](#models)
- [Installation and environments](#installation-and-environments)
- [Good practice](#good-practice)
- [Adding a model](#adding-a-model)
- [Licence](#licence)

## Quick start

```python
import origami as ori
import pandas as pd

cat = pd.read_csv("catalog.csv")                       # one row per object: id, period, target, ...
lcs = ori.LightCurves.from_csv_dir("light_curves/",    # one CSV per object: time, mag, err
                                   ids=cat["id"], err_col="mag_err", survey="ztf", band="r")
lcs = lcs.with_periods(cat["period"])                  # needed by folded inputs and Fourier features

reps = {
    "chronos":  ori.Embedder("chronos-bolt-small", input="folded-grid"),
    "mantis":   ori.Embedder("mantis-v2", input="folded-grid"),
    "fourier":  ori.HandCrafted("fourier"),
    "physical": ori.HandCrafted("physical"),
    "period":   ori.Catalog(cat, ["period"]),
}
res = ori.compare(lcs, cat["target"], reps, task="regression",
                  combine=[("chronos", "physical"), ("mantis", "physical"), ("fourier", "period")],
                  subsample_epochs=["all", 50, 20], seeds=[0, 1, 2])

print(res.table())     # primary metric (RMSE): rows = representations, columns = epoch levels
res.records            # every run with all metrics, as a DataFrame
```

## Concepts

### Light curves
`LightCurves` holds a set of single-band curves. Each curve has:
- its own time, magnitude and error arrays (any length);
- an identifier;
- a provenance tag (`survey`, `band`);
- optionally a period, a reference epoch and the period's origin (`"catalog"` or `"estimated"`).

| constructor | input |
|---|---|
| `LightCurves.from_csv_dir(dir, ids=..., ...)` | one CSV per object; magnitudes with errors, or fluxes (errors are then converted) |
| `LightCurves.from_frame(df, id_col=..., ...)` | a long table, one row per observation |
| `LightCurves.from_arrays(times, mags, errs, ...)` | lists of arrays |

Other operations:
- `lcs.with_periods(periods, epochs=None)` attaches periods.
- `lcs.subsample(k, seed)` keeps k random epochs per curve. The selection is reproducible and
  does not depend on the other curves in the set.

### Input transforms
How a curve becomes the sequence a model sees. Set it per embedder with `input=`.

| `input` | sequence | needs a period |
|---|---|---|
| `raw` | magnitudes in time order, z-normalised, with real timestamps | no |
| `folded` | magnitudes sorted by phase, z-normalised | yes |
| `folded-grid` | phase-folded curve resampled on a regular grid (default 200 points), z-normalised, maximum light at phase 0 | yes |
| `padded-starembed` | last 200 points, zero left-padding: the protocol of [StarEmbed](https://arxiv.org/abs/2510.06200), for comparison | no |

How `folded-grid` resamples the curve:
- It uses a smoothing spline, with automatic smoothing chosen by GCV.
- If that fit collapses to a near-linear curve, it falls back to a Fourier series whose
  number of harmonics is chosen by BIC.
- The curve is aligned on maximum light, so the sequence does not depend on the reference epoch.

Every transform also passes, for each curve, the times and the magnitude scale removed by
z-normalisation. Models that use timestamps receive the times; models trained on unscaled
magnitudes get the scale back.

### Representations
Anything with a `name` and a `transform(lcs) -> array (n_curves, dim)` method.

| class | what it gives |
|---|---|
| `Embedder(model, input=...)` | frozen foundation-model embedding: encoder states mean-pooled over the positions that hold real data; cached on disk |
| `HandCrafted("light-curve")` | about 30 period-free statistics from the [`light-curve`](https://github.com/light-curve/light-curve-python) package |
| `HandCrafted("fourier", n_harmonics=3)` | Fourier amplitudes and cos/sin of φ21, φ31 on the folded curve |
| `HandCrafted("physical")` | log period and folded peak-to-peak amplitude |
| `Catalog(table, columns)` | columns of an external table, row-aligned with the curves |
| your own object | e.g. an autoencoder: anything implementing `transform(lcs)` |

`combine=[("a", "b"), ...]` evaluates concatenations of representations. In distance-based
tasks each concatenated block is weighted by 1/√width, so a narrow feature block is not
outweighed by a wide embedding.

### Tasks and metrics
| `task` | probe | metrics |
|---|---|---|
| `regression` | standardised ridge regression (RidgeCV) | RMSE, MAE |
| `classification` | standardised logistic regression with balanced class weights | balanced accuracy, macro F1 |
| `clustering` | k-means (k = number of classes), HDBSCAN, or [CLiMB](https://github.com/LorenzoMonti/CLiMB) | ARI, NMI |
| `anomaly` | kNN distance or isolation forest, fitted on the normal objects of the training folds (`y` = 1 for anomalies) | AUROC, average precision |

The probes are deliberately simple and fixed, so that results measure the representation
and not the model on top of it.

Evaluation options of `compare`:
- **`folds`**: defaults to 5-fold cross-validation, stratified for classification and
  anomaly. Pass your own fold index to reuse an existing protocol.
- **`split=(train_mask, test_mask)`**: train on one subset and score another, e.g. bright
  versus faint, or one survey versus another.
- **`groups=labels`**: also report metrics per group, e.g. magnitude bins or classes. Read
  them with `res.group_table()`.
- **`subsample_epochs` and `seeds`**: thin every curve to the given numbers of epochs,
  identically for all representations.
- **Task options** are passed through, e.g. `method="hdbscan"`, `detector="isolation-forest"`
  or `pca=20`.

## Command-line workflow

For large runs, especially on clusters whose compute nodes have no internet access, an
experiment is described by one YAML file:

```yaml
name: my_experiment
output: /path/to/runs/my_experiment
data:
  light_curves: /path/to/light_curves     # one CSV per object, "{id}.csv"
  catalog: /path/to/catalog.csv           # one row per object
  id_col: id
  time_col: time
  mag_col: mag
  err_col: mag_err                        # or flux_col + flux_err_col
  survey: ztf
  band: r
  period_col: period                      # optional
  epoch_col: epoch                        # optional
  fold_col: fold                          # optional, otherwise k-fold
representations:
  bolt:     {model: chronos-bolt-small, input: folded-grid}
  mantis:   {model: mantis-v2, input: folded-grid}
  moirai:   {model: moirai-2.0-small, input: folded-grid, setup: "source ~/venvs/origami-moirai/bin/activate"}
  astromer: {model: astromer2-macho, input: raw}
  fourier:  {features: fourier}
  physical: {features: physical}
  period:   {catalog: [period]}
  my_model: {python: "my_package.encoders:MyEncoder", path: /path/to/my_package, kwargs: {checkpoint: model.pt}, version: v1}
combine: [[bolt, physical], [mantis, physical], [fourier, period]]
tasks:
  - {task: regression, target: my_target}
  - {task: classification, target: class}
  - task: classification
    target: class
    split: {train: "mag < 19", test: "mag >= 19"}       # pandas queries, or {column, train, test}
    groups: {column: mag, bins: [16, 18, 19, 20, 21]}
  - {task: clustering, target: class, method: hdbscan}
subsample_epochs: [all, 50, 20]
seeds: [0, 1, 2]
```

```bash
origami models                                   # available models (licence, size, install, installed?)
origami prefetch exp.yaml                        # on a machine with internet: download all weights
origami slurm exp.yaml --account ACC --partition PART \
      --setup "module load python" --setup "source ~/venvs/origami/bin/activate"
sbatch origami_my_experiment.sbatch              # one array task per model, each runs `origami embed`
origami status exp.yaml                          # which embeddings are stored / missing
origami collect exp.yaml                         # run all tasks, write results/*.csv and summary.md
```

What each step does:
- **`origami embed exp.yaml --representation NAME`** computes one representation for every
  epoch level and seed. It also works without SLURM.
- **Embeddings** are stored under `output/embeddings`. Their key covers the model and its
  settings, the input transform, the code version and the data, including subsampling and seed.
- **`collect` never runs a model.** If an embedding is missing, it says which representation
  to rerun.
- **User models** (`python:`) are computed and cached like the foundation models. Change
  `version` whenever your weights change.
- **`setup:`** gives a representation a shell line run by its own array task, for models
  that need a separate environment.
- **Offline nodes:** the generated job sets `HF_HUB_OFFLINE=1`.

## Models

`origami models` prints the live list and whether each package is installed.

| family | checkpoints | params | licence of the weights | uses timestamps | extra |
|---|---|---|---|---|---|
| [Chronos](https://github.com/amazon-science/chronos-forecasting) (T5) | `chronos-t5-tiny/mini/small/base` | 8M–200M | Apache-2.0 | no | `chronos` |
| Chronos-Bolt | `chronos-bolt-tiny/small` | 9M–48M | Apache-2.0 | no | `chronos` |
| Chronos-2 | `chronos-2-small`, `chronos-2` | 28M, 120M | Apache-2.0 | no | `chronos` |
| [Moirai](https://github.com/SalesforceAIResearch/uni2ts) | `moirai-1.1-small/base`, `moirai-2.0-small` | 11M–91M | **CC-BY-NC-4.0 (non-commercial)** | no | `moirai` |
| [MOMENT](https://huggingface.co/AutonLab/MOMENT-1-small) | `moment-1-small/base/large` | 38M–346M | MIT | no | `moment` |
| [Mantis](https://huggingface.co/paris-noah/Mantis-8M) | `mantis-8m`, `mantis-plus`, `mantis-v2`, `mantis-v2-cls` | 4M–8M | Apache-2.0 | no | `mantis` |
| [Toto 2.0](https://github.com/DataDog/toto) | `toto-2.0-4m/22m/313m/1b/2.5b` | 4M–2.5B | Apache-2.0 | no | `toto` |
| [Timestral](https://github.com/mistralai/timestral) | `timestral-5m/28m/123m/209m` | 5M–209M | **Modified MIT: excludes companies above $20M monthly revenue** | no | `timestral` |
| [Astromer](https://github.com/astromer-science) | `astromer1-macho`, `astromer2-macho`, `astromer1-ztfdr20` | 0.7M–4M | CC-BY-4.0; **GPL-3.0 for `astromer1-ztfdr20`** | **yes** | `astromer` |
| [StarCLR](https://github.com/dj-y/StarCLR-Inference) | `starclr` | 19M | MIT | **yes** | `starclr` |

Notes on specific families:
- **Astromer** runs through the ONNX exports of the
  [light-curve](https://github.com/light-curve/prep-models) project, so it needs neither
  TensorFlow nor the original package.
- **StarCLR** uses a PyTorch re-implementation of the official inference code. Its weights
  come from [Zenodo](https://doi.org/10.5281/zenodo.19728042), are verified by checksum,
  and are stored in `$ORIGAMI_WEIGHTS`.
- **Toto, Timestral and Moirai** have no public embedding API. Their states are read from
  internal modules, stated in each plugin's docstring, so their package versions are pinned.
- **Not included yet:**
  - models that require `trust_remote_code`;
  - models without public pretrained weights.

**The weights are not part of `origami`.** They are downloaded from their original
sources, and their licences govern your use of them. Check the non-commercial,
copyleft and revenue clauses marked above before using those models outside academic
research.

## Installation and environments

```bash
pip install "origami-astro[features,chronos,mantis,astromer,starclr]"
```

| extra | adds |
|---|---|
| `features` | hand-crafted features (`light-curve`) |
| `chronos`, `mantis`, `astromer`, `starclr`, `timestral` | those model families (they can share one environment) |
| `moirai`, `moment`, `toto` | those model families, **each in its own environment** (see below) |
| `dev` | tests (`pytest`) |

Some model packages pin incompatible versions:

| environment | extras | reason |
|---|---|---|
| main | `features, chronos, mantis, astromer, starclr, timestral` | — |
| moirai | `features, moirai` | `uni2ts` 2.0 pins `torch<2.5` and `numpy~=1.26` |
| moment | `features, moment` | `momentfm` pins `transformers==4.33.3` and `numpy==1.25.2` |
| toto | `features, toto` | `toto2` needs Python ≥ 3.12 |

In the YAML workflow, point each representation to its environment with `setup:`.

Timestral is not on PyPI; install it from GitHub:
```bash
pip install "timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
```

On clusters, build environments on top of the site's Python/PyTorch module
(`python -m venv --system-site-packages ...`), so that only the model packages are added.

Environment variables:
- `ORIGAMI_CACHE`: embedding cache in interactive use (the CLI stores embeddings under `output/embeddings`).
- `ORIGAMI_WEIGHTS`: where weights not hosted on Hugging Face are stored.
- `HF_HUB_OFFLINE=1`: use only weights that are already downloaded.

Requires Python ≥ 3.10.

## Good practice

1. **Compare input transforms, not only models.** For periodic sources, folded and raw
   inputs can give very different results with the same model. Each model also expects data
   that looks like its pre-training data: scale, flux versus magnitude, regular versus
   irregular sampling.
2. **Put back what normalisation removes.** Z-normalised, folded inputs carry no period and
   no amplitude. Combine embeddings with `HandCrafted("physical")` or catalogue values
   before comparing them with features that contain this information.
3. **Mind collapsed embeddings.** Some models map all objects to nearly parallel vectors,
   with the information in small residual directions. Standardised probes, which `origami`
   always uses, recover it; raw distances do not.
4. **Do not assume larger checkpoints are better.** Compare sizes within a family.
5. **Report variability.** Use several `seeds` and fixed `folds`, and check that differences
   between representations exceed the fold-to-fold scatter before ranking them.
6. **Check where the labels come from.** If the target was derived from light-curve
   features (e.g. a quantity computed from Fourier parameters), a representation can score
   well by re-learning that relation. Validate on independently measured labels when possible.

## Adding a model

Add one file to `origami/models/`; it is discovered automatically.

```python
MODELS = {"my-model": dict(family="my", hf_id="org/my-model", license="MIT", extra="my",
                           context=512, params="10M", uses_time=False)}

class Encoder:
    def __init__(self, spec, device=None):
        ...  # heavy imports here, not at module level

    def encode(self, values, masks, times, batch_size=256, scales=None):   # `scales` is optional
        ...  # return (n_curves, dim): states mean-pooled over positions where mask is True
```

Add a test in `tests/test_model_<name>.py` that checks:
- shape, finite values and determinism;
- that a curve's embedding does not change when it is batched with longer curves.

## Licence

`origami` is released under the [MIT licence](https://github.com/LorenzoMonti/origami/blob/main/LICENSE). Model weights are downloaded from
their original sources and keep their own licences (see [Models](#models)).

If you use `origami` in your work, please cite it (GitHub's *Cite this repository* button
uses [`CITATION.cff`](https://github.com/LorenzoMonti/origami/blob/main/CITATION.cff)) and
the papers of the models you use.
