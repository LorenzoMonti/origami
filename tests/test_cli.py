import numpy as np
import pandas as pd
import pytest

from origami.cli import main


def write_dataset(root, n=60, seed=0):
    rng = np.random.default_rng(seed)
    lc_dir = root / "lc"
    lc_dir.mkdir()
    rows = []
    for i in range(n):
        cls = i % 2
        p = rng.uniform(0.45, 0.7) if cls == 0 else rng.uniform(0.25, 0.4)
        t = np.sort(rng.uniform(0, 1000, 50))
        ph = (t / p) % 1
        m = 15 + (0.3 * np.cos(2 * np.pi * ph) if cls == 0 else 0.4 * (ph - 0.5)) + rng.normal(0, 0.01, 50)
        pd.DataFrame(dict(time=t, mag=m, mag_err=np.full(50, 0.01))).to_csv(lc_dir / f"star{i}.csv", index=False)
        rows.append(dict(source_id=f"star{i}", P=p, epoch=0.0, cls=cls, target=-1 - 2 * p + rng.normal(0, 0.01)))
    pd.DataFrame(rows).assign(mag=lambda d: np.linspace(15, 21, len(d))).to_csv(root / "catalog.csv", index=False)


def write_config(root, representations, combine="[]"):
    cfg = root / "exp.yaml"
    cfg.write_text(f"""
name: synth
output: {root / "run"}
data:
  light_curves: {root / "lc"}
  catalog: {root / "catalog.csv"}
  id_col: source_id
  err_col: mag_err
  survey: synthetic
  band: G
  period_col: P
  epoch_col: epoch
representations:
{representations}
combine: {combine}
tasks:
  - {{task: regression, target: target}}
  - {{task: classification, target: cls}}
  - {{task: clustering, target: cls, method: kmeans}}
subsample_epochs: [all, 30]
seeds: [0, 1]
""")
    return cfg


def test_collect_with_features_only(tmp_path, capsys):
    write_dataset(tmp_path)
    cfg = write_config(tmp_path, "  fourier: {features: fourier}\n  physical: {features: physical}\n  P: {catalog: [P]}",
                       combine="[[fourier, P]]")
    main(["collect", str(cfg)])
    out = tmp_path / "run" / "results"
    assert (out / "summary.md").exists()
    table = pd.read_csv(out / "regression_target_table.csv", index_col=0)
    assert set(table.index) == {"fourier", "physical", "P", "fourier + P"}
    assert list(table.columns) == ["all", "30"]


def test_config_validation(tmp_path):
    write_dataset(tmp_path)
    cfg = write_config(tmp_path, "  bad: {features: fourier, catalog: [P]}")
    with pytest.raises(ValueError, match="exactly one"):
        main(["collect", str(cfg)])


def test_embed_status_collect_slurm(tmp_path, capsys):
    pytest.importorskip("chronos")
    write_dataset(tmp_path, n=40)
    cfg = write_config(tmp_path, "  chronos: {model: chronos-t5-tiny, input: folded-grid, setup: 'source env_a'}\n"
                                 "  P: {catalog: [P]}",
                       combine="[[chronos, P]]")
    with pytest.raises(SystemExit):  # collect must not run models
        main(["collect", str(cfg)])
    main(["status", str(cfg)])
    assert "0/3 stored" in capsys.readouterr().out
    main(["embed", str(cfg), "--representation", "chronos", "--device", "cpu"])
    main(["status", str(cfg)])
    assert "3/3 stored" in capsys.readouterr().out
    main(["collect", str(cfg)])
    table = pd.read_csv(tmp_path / "run" / "results" / "classification_cls_table.csv", index_col=0)
    assert "chronos + P" in table.index
    sbatch = tmp_path / "job.sbatch"
    main(["slurm", str(cfg), "--account", "acc", "--partition", "gpu", "--setup", "module load x", "--out", str(sbatch)])
    text = sbatch.read_text()
    assert "--array=0-0" in text and "REPRESENTATIONS=(chronos)" in text and "module load x" in text
    assert "HF_HUB_OFFLINE=1" in text and "SETUPS=('source env_a')" in text


class ToyEncoder:
    """Stand-in for a user model (e.g. a VAE): mean and std of each curve."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def transform(self, lcs):
        return np.array([[c.mag.mean() * self.scale, c.mag.std()] for c in lcs])


def test_python_representation_split_groups(tmp_path):
    write_dataset(tmp_path)
    cfg = write_config(tmp_path, "  toy: {python: 'test_cli:ToyEncoder', kwargs: {scale: 2.0}}\n  P: {catalog: [P]}",
                       combine="[[toy, P]]")
    text = cfg.read_text().replace("tasks:\n", "tasks:\n  - {task: classification, target: cls, split: {train: 'mag < 18', "
                                   "test: 'mag >= 18'}, groups: {column: mag, bins: [14, 18, 20, 22]}}\n")
    cfg.write_text(text)
    with pytest.raises(SystemExit):  # python representations are computed by embed jobs, not by collect
        main(["collect", str(cfg)])
    main(["embed", str(cfg), "--representation", "toy"])
    main(["collect", str(cfg)])
    out = tmp_path / "run" / "results"
    split = pd.read_csv(out / "classification_cls_split_groups_table.csv", index_col=0)
    assert "toy + P" in split.index
    groups = pd.read_csv(out / "classification_cls_split_groups_groups_epall.csv", index_col=0)
    assert list(groups.columns) == ["(18, 20]", "(20, 22]"]  # only test-set groups are scored
