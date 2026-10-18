"""Cell loop with on-disk caching and resume.

  baseline: one cell per (task, length), unmodified model          -> D5, gate G2
  sweep:    band=none + one cell per band per (task, length)        -> D1, gate G4

Every cell is written to its own JSON immediately; existing cells are skipped
unless --force.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.config import config_hash, lib_versions, load_config, model_slug, write_json
from src.eval.metrics import accuracy, bootstrap_ci
from src.model_io import generate, load_model, model_revision
from src.rope.bands import band_label, band_slices, band_wavelengths
from src.rope.patch import RopePatcher, band_transform
from src.tasks import get_task

HASH_KEYS = ["model_id", "revision", "dtype", "load_in_4bit", "seed", "depth_fraction",
             "length_tolerance", "filler", "task_params"]


def stage_dir(cfg, stage, mode=None, factor=None):
    root = Path(cfg["results_dir"]) / model_slug(cfg["model_id"])
    if stage == "baseline":
        return root / "baseline"
    return root / "sweep" / ("zero" if mode == "zero" else f"compress_f{factor:g}")


def cell_path(d, task, length, band=None, stage="sweep"):
    if stage == "baseline":
        return d / f"{task}_{length}.json"
    return d / f"{task}_{length}_band{'none' if band is None else band}.json"


def example_seeds(cfg, n):
    return [cfg["seed"] * 1000 + k for k in range(n)]


def gold_repr(gold):
    return gold["top3"] if isinstance(gold, dict) and "top3" in gold else gold


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg["seed"])
        self.model, self.tok = load_model(cfg)
        self.patcher = RopePatcher(self.model)
        inv = self.patcher.inv_freq
        self.rotary_dim = 2 * len(inv)
        self.slices = band_slices(self.rotary_dim, cfg["n_bands"])
        self.wavelengths = band_wavelengths(inv.numpy(), self.slices)
        max_pos = self.model.config.max_position_embeddings
        for L in cfg["lengths"]:
            if L > max_pos:
                raise ValueError(f"length {L} exceeds native context {max_pos}")
        self._examples = {}
        self.meta = {
            "model_id": cfg["model_id"],
            "model_revision": model_revision(self.model),
            "dtype": str(next(self.model.parameters()).dtype),
            "versions": lib_versions(),
            "config_hash": config_hash(cfg, HASH_KEYS),
            "seed": cfg["seed"],
            "filler": cfg["filler"],
            "depth_fraction": cfg["depth_fraction"],
            "rotary_dim": self.rotary_dim,
            "n_bands": cfg["n_bands"],
        }

    def examples(self, task_name, length, n):
        key = (task_name, length, n)
        if key not in self._examples:
            task = get_task(task_name, self.cfg.get("task_params"))
            exs = [task.generate(self.tok, length, s, self.cfg["depth_fraction"],
                                 self.cfg["length_tolerance"])
                   for s in example_seeds(self.cfg, n)]
            self._examples[key] = (task, exs)
        return self._examples[key]

    def run_cell(self, task_name, length, n, band=None, mode=None, factor=None):
        task, exs = self.examples(task_name, length, n)
        if band is None:
            self.patcher.restore()
        else:
            self.patcher.apply(band_transform(self.rotary_dim, self.cfg["n_bands"], band, mode, factor),
                               label=f"band{band}")
        t0 = time.time()
        records = []
        try:
            for i, ex in enumerate(exs):
                out = generate(self.model, self.tok, ex.prompt, task.max_new_tokens)
                records.append({"i": i, "correct": bool(task.score(out, ex.gold)),
                                "pred": out[:300], "gold": gold_repr(ex.gold),
                                "n_tokens": ex.n_tokens})
        finally:
            self.patcher.restore()
        correct = [r["correct"] for r in records]
        bs = self.cfg["bootstrap"]
        return {
            **self.meta,
            "task": task_name, "task_params": task.params, "length": length, "n_examples": n,
            "band": band, "mode": mode, "compress_factor": factor,
            "band_dims": None if band is None else [self.slices[band].start, self.slices[band].stop],
            "wavelength_range": None if band is None else self.wavelengths[band],
            "accuracy": accuracy(correct),
            "ci95": bootstrap_ci(correct, bs["n_resamples"], bs["ci"], self.cfg["seed"]),
            "wall_seconds": round(time.time() - t0, 2),
            "examples": records,
        }


def run(cfg, stage, tasks, lengths, n, bands=None, mode=None, factor=None, force=False):
    runner = None
    chash = config_hash(cfg, HASH_KEYS)
    d = stage_dir(cfg, stage, mode, factor)
    band_list = [None] if stage == "baseline" else [None] + list(bands)
    out = []
    for L in lengths:
        for t in tasks:
            for b in band_list:
                p = cell_path(d, t, L, b, stage)
                if p.exists() and not force:
                    res = json.loads(p.read_text())
                    if res["config_hash"] != chash or res["n_examples"] != n:
                        raise SystemExit(f"{p} was made with a different config "
                                         f"(hash {res['config_hash']} vs {chash}, n {res['n_examples']} vs {n}). "
                                         "Use another --results-dir, or --force to overwrite.")
                    print(f"skip  {p}  acc={res['accuracy']:.3f}")
                else:
                    runner = runner or Runner(cfg)
                    res = runner.run_cell(t, L, n, b, mode, factor)
                    write_json(p, res)
                    lo, hi = res["ci95"]
                    print(f"done  {p}  acc={res['accuracy']:.3f} [{lo:.2f},{hi:.2f}] "
                          f"{res['wall_seconds']}s")
                out.append(res)
    return out


def summarize(results):
    rows = [{"task": r["task"], "length": r["length"],
             "band": "none" if r["band"] is None else r["band"],
             "acc": r["accuracy"], "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1],
             "n": r["n_examples"]} for r in results]
    return pd.DataFrame(rows)


def gate_g2(cfg, df):
    L0 = min(df["length"])
    ok = True
    for task, thr in cfg["gate_g2"].items():
        sub = df[(df.task == task) & (df.length == L0)]
        if sub.empty:
            print(f"G2 {task}@{L0}: not run")
            continue
        acc = float(sub.acc.iloc[0])
        passed = acc > thr
        ok &= passed
        print(f"G2 {task}@{L0}: acc={acc:.3f} > {thr}  {'PASS' if passed else 'FAIL'}")
    print("GATE G2:", "PASS" if ok else "FAIL")
    return ok


def gate_g4(cfg, results):
    bdir = stage_dir(cfg, "baseline")
    ok = True
    for r in results:
        if r["band"] is not None:
            continue
        p = cell_path(bdir, r["task"], r["length"], stage="baseline")
        if not p.exists():
            print(f"G4 {r['task']}@{r['length']}: no baseline cell")
            continue
        base = json.loads(p.read_text())
        same = [a["correct"] for a in base["examples"]] == [b["correct"] for b in r["examples"]]
        ok &= same
        print(f"G4 {r['task']}@{r['length']}: baseline={base['accuracy']:.3f} "
              f"band=none={r['accuracy']:.3f}  {'PASS' if same else 'FAIL'}")
    print("GATE G4:", "PASS" if ok else "FAIL")
    return ok


def init_clearml(cfg, a):
    """Optional ClearML tracking. With --clearml-queue the script re-launches on that queue."""
    from clearml import Task

    cm = cfg.get("clearml", {})
    name = f"{cm.get('task_name', 'rope_freq_ablation')}_{a.stage}"
    task = Task.init(project_name=cm.get("project", "Blue-2.0"), task_name=name,
                     reuse_last_task_id=False, auto_connect_frameworks=False)
    task.connect_configuration(cfg, name="config")
    if a.clearml_queue:
        task.execute_remotely(queue_name=a.clearml_queue, exit_process=True)
    return task


def report_clearml(task, cfg, stage, results, df, mode, factor):
    log = task.get_logger()
    for r in results:
        band = "none" if r["band"] is None else f"band{r['band']}"
        log.report_scalar(f"acc/{r['task']}_{r['length']}", band, r["accuracy"],
                          iteration=-1 if r["band"] is None else r["band"])
    log.report_table("summary", stage, iteration=0, table_plot=df)
    task.upload_artifact(f"results_{stage}", artifact_object=str(stage_dir(cfg, stage, mode, factor)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--stage", choices=["baseline", "sweep"], required=True)
    ap.add_argument("--model-id")
    ap.add_argument("--results-dir")
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--lengths", nargs="+", type=int)
    ap.add_argument("--n", type=int, help="examples per cell")
    ap.add_argument("--bands", nargs="+", type=int, help="subset of bands (sweep)")
    ap.add_argument("--mode", choices=["compress", "zero"])
    ap.add_argument("--factor", type=float)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VAL",
                    help="dotted config override, e.g. task_params.fwe.alpha=3.0")
    ap.add_argument("--no-gate", action="store_true", help="do not exit non-zero on a failed gate")
    ap.add_argument("--clearml", action="store_true", help="track this run in ClearML")
    ap.add_argument("--clearml-queue", help="enqueue on this ClearML queue instead of running here")
    a = ap.parse_args()

    cfg = load_config(a.config, {"model_id": a.model_id, "results_dir": a.results_dir}, a.set)
    tasks = a.tasks or cfg["tasks"]
    lengths = a.lengths or cfg["lengths"]
    n = a.n or cfg["n_examples"]
    mode = a.mode or cfg["intervention"]["mode"]
    factor = a.factor or cfg["intervention"]["compress_factor"]
    bands = a.bands if a.bands is not None else range(cfg["n_bands"])
    cm_task = init_clearml(cfg, a) if (a.clearml or a.clearml_queue) else None

    results = run(cfg, a.stage, tasks, lengths, n, bands, mode, factor, a.force)
    df = summarize(results)
    print("\n" + df.to_string(index=False))
    if a.stage == "baseline":
        out = stage_dir(cfg, "baseline") / "baseline_table.csv"
        every = [json.loads(p.read_text()) for p in sorted(out.parent.glob("*_*.json"))]
        summarize(every).sort_values(["length", "task"]).to_csv(out, index=False)  # all lengths, not just this run
        print(f"saved -> {out}\n")
        gate_ok = gate_g2(cfg, df)
    else:
        print()
        gate_ok = gate_g4(cfg, results)
    if cm_task:
        report_clearml(cm_task, cfg, a.stage, results, df, mode, factor)
        cm_task.close()
    if not gate_ok and not a.no_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
