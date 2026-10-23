"""
Command-line workflow for clusters with offline compute nodes.

  origami models                               list the available models (licence, size, pip extra, installed?)
  origami prefetch config.yaml                 download the weights of every model in the config
                                               (run on a node with internet, e.g. a login node)
  origami slurm config.yaml --account A ...    write an sbatch job array, one model representation per task
  origami embed config.yaml --representation R compute and store the embeddings of R for every epoch
                                               level and seed (what each array task runs)
  origami status config.yaml                   which embeddings are stored and which are missing
  origami collect config.yaml                  run the tasks on stored embeddings, write result tables
"""
import argparse
import json
import os
import sys
import time

from .config import embedding_jobs, epochs_label, load_config
from .embedders import MissingEmbedding

SBATCH = """#!/bin/bash
#SBATCH --job-name=origami_{name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-{last}
#SBATCH --output={logs}/%x_%A_%a.out

REPRESENTATIONS=({reps})
SETUPS=({setups})
{setup}
eval "${{SETUPS[$SLURM_ARRAY_TASK_ID]}}"
export HF_HUB_OFFLINE=1
origami embed {config} --representation ${{REPRESENTATIONS[$SLURM_ARRAY_TASK_ID]}}
"""


def cmd_models(args):
    import importlib.util

    from .embedders import available_models
    probe = dict(chronos="chronos", moirai="uni2ts", moment="momentfm", mantis="mantis", astromer="onnxruntime",
                 toto="toto2", timestral="timestral")
    rows = []
    for name, spec in sorted(available_models().items()):
        mod = probe.get(spec.get("extra"), spec.get("extra"))
        installed = mod is not None and importlib.util.find_spec(mod) is not None
        if args.installed and not installed:
            continue
        rows.append((name, spec.get("family", ""), str(spec.get("params", "")), spec.get("license", ""),
                     "yes" if spec.get("uses_time") else "no", f"origami-astro[{spec.get('extra')}]",
                     "yes" if installed else "no"))
    head = ("model", "family", "params", "licence", "times", "install", "installed")
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def cmd_prefetch(exp, args):
    from .embedders import prefetch
    for m in exp.models():
        print(f"downloading {m}", flush=True)
        prefetch([m])
    print("done; on compute nodes set HF_HUB_OFFLINE=1")


def cmd_embed(exp, args):
    lcs, cat = exp.load()
    rep = exp.representation(args.representation, cat, compute=True, device=args.device)
    if args.representation not in exp.model_representations():
        sys.exit(f"{args.representation!r} is not a model representation: nothing to embed")
    for name, level, seed in embedding_jobs(exp):
        if name != args.representation:
            continue
        t0 = time.time()
        emb = rep.transform(lcs.subsample(level, seed=seed))
        print(f"{name} epochs={epochs_label(level)} seed={seed} shape={emb.shape} {time.time() - t0:.0f}s",
              flush=True)


def _stored(exp, lcs, cat):
    rows = []
    for name, level, seed in embedding_jobs(exp):
        rep = exp.representation(name, cat, compute=False)
        path = rep._cache_path(lcs.subsample(level, seed=seed))
        rows.append((name, epochs_label(level), seed, os.path.exists(path)))
    return rows


def cmd_status(exp, args):
    lcs, cat = exp.load()
    rows = _stored(exp, lcs, cat)
    for name, level, seed, ok in rows:
        print(f"{'ok     ' if ok else 'MISSING'} {name} epochs={level} seed={seed}")
    missing = sorted({r[0] for r in rows if not r[3]})
    print(f"\n{sum(r[3] for r in rows)}/{len(rows)} stored" + (f"; rerun: {' '.join(missing)}" if missing else ""))


def cmd_collect(exp, args):
    try:
        results = exp.run_tasks(compute=False)
    except MissingEmbedding as e:
        sys.exit(f"{e}\nrun `origami status {args.config}` to see what is missing")
    os.makedirs(exp.results_dir, exist_ok=True)
    summary = []
    for task, target, kwargs, res in results:
        tag = "_".join([task, target] + [f"{k}-{v}" for k, v in sorted(kwargs.items())
                                         if k not in ("split", "groups")])
        tag += ("_split" if "split" in kwargs else "") + ("_groups" if "groups" in kwargs else "")
        res.records.assign(task=task, target=target).to_csv(os.path.join(exp.results_dir, f"{tag}.csv"),
                                                            index=False)
        table = res.table()
        table.to_csv(os.path.join(exp.results_dir, f"{tag}_table.csv"))
        summary.append(f"## {task} — {target} {kwargs or ''}\n\n{table.round(4).to_markdown()}\n")
        print(f"\n{task} on {target} {kwargs or ''}\n{table.round(4).to_string()}")
        if (res.records.group != "all").any():
            for level in exp.subsample_epochs:
                gt = res.group_table(epochs=level)
                gt.to_csv(os.path.join(exp.results_dir, f"{tag}_groups_ep{epochs_label(level)}.csv"))
                summary.append(f"### per group, epochs={epochs_label(level)}\n\n{gt.round(4).to_markdown()}\n")
                print(f"per group, epochs={epochs_label(level)}\n{gt.round(4).to_string()}")
    with open(os.path.join(exp.results_dir, "summary.md"), "w") as f:
        f.write(f"# {exp.name}\n\nprovenance: {results[0][3].provenance if results else ''}\n\n" + "\n".join(summary))
    with open(os.path.join(exp.results_dir, "config_used.json"), "w") as f:
        json.dump(dict(representations=exp.representations, combine=exp.combine, tasks=exp.tasks,
                       subsample_epochs=exp.subsample_epochs, seeds=exp.seeds, data=exp.data), f, indent=2)
    print(f"\nresults in {exp.results_dir}")


def cmd_slurm(exp, args):
    reps = exp.model_representations()
    if not reps:
        sys.exit("no model representations in the config: nothing to submit")
    logs = os.path.join(exp.output, "logs")
    os.makedirs(logs, exist_ok=True)
    import shlex
    setups = " ".join(shlex.quote(exp.representations[r].get("setup", "")) for r in reps)
    script = SBATCH.format(name=exp.name, account=args.account, partition=args.partition, cpus=args.cpus,
                           mem=args.mem, time=args.time, last=len(reps) - 1, logs=logs, reps=" ".join(reps),
                           setups=setups,
                           setup="\n".join(args.setup or []), config=os.path.abspath(args.config))
    path = args.out or f"origami_{exp.name}.sbatch"
    with open(path, "w") as f:
        f.write(script)
    print(f"wrote {path} ({len(reps)} array tasks: {', '.join(reps)})\nsubmit with: sbatch {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="origami", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("models").add_argument("--installed", action="store_true", help="only models whose package is installed")
    for name in ("prefetch", "status", "collect"):
        sub.add_parser(name).add_argument("config")
    p = sub.add_parser("embed")
    p.add_argument("config")
    p.add_argument("--representation", required=True)
    p.add_argument("--device", default=None)
    p = sub.add_parser("slurm")
    p.add_argument("config")
    p.add_argument("--account", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="60G")
    p.add_argument("--setup", action="append", help="shell line run before origami (repeatable), "
                                                    "e.g. --setup 'module load python'")
    p.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.command == "models":
        return cmd_models(args)
    exp = load_config(args.config)
    dict(prefetch=cmd_prefetch, embed=cmd_embed, status=cmd_status, collect=cmd_collect,
         slurm=cmd_slurm)[args.command](exp, args)


if __name__ == "__main__":
    main()
