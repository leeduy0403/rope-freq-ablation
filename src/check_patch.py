"""Gate G3: prove the inv_freq patch is correct before any sweep.

  1. identity transform  -> logits identical to unpatched
  2. compress one band   -> logits change (forward really reads the buffer)
  3. zero all freqs      -> large accuracy drop (model position-blind)
  4. restore             -> logits and accuracy identical to the original
"""
import argparse

import torch

from src.config import load_config, model_slug, write_json
from src.eval.metrics import accuracy
from src.model_io import generate
from src.eval.runner import Runner
from src.rope.patch import band_transform, identity, zero_all

PROBE = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"


@torch.no_grad()
def logits(model, tok, text):
    enc = tok(text, return_tensors="pt").to(model.device)
    return model(**enc).logits.float().cpu()


def task_acc(r, task_name, length, n):
    task, exs = r.examples(task_name, length, n)
    return accuracy([task.score(generate(r.model, r.tok, e.prompt, task.max_new_tokens), e.gold)
                     for e in exs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model-id")
    ap.add_argument("--results-dir")
    ap.add_argument("--tasks", nargs="+", default=["s_niah"])
    ap.add_argument("--length", type=int, help="default: shortest configured length")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VAL")
    a = ap.parse_args()
    cfg = load_config(a.config, {"model_id": a.model_id, "results_dir": a.results_dir}, a.set)
    L = a.length or min(cfg["lengths"])
    r = Runner(cfg)
    P, m, tok = r.patcher, r.model, r.tok
    rep = {**r.meta, "n": a.n, "length": L, "tests": {}}

    ref = logits(m, tok, PROBE)
    P.apply(identity, "identity")
    d_id = (logits(m, tok, PROBE) - ref).abs().max().item()
    P.restore()
    rep["tests"]["identity_max_abs_diff"] = d_id

    P.apply(band_transform(r.rotary_dim, cfg["n_bands"], 0, "compress", 8.0), "band0")
    d_b0 = (logits(m, tok, PROBE) - ref).abs().max().item()
    P.restore()
    rep["tests"]["band0_compress_max_abs_diff"] = d_b0

    base, zero, restored = {}, {}, {}
    for t in a.tasks:
        base[t] = task_acc(r, t, L, a.n)
        P.apply(zero_all, "zero_all")
        zero[t] = task_acc(r, t, L, a.n)
        P.restore()
        restored[t] = task_acc(r, t, L, a.n)
    d_rest = (logits(m, tok, PROBE) - ref).abs().max().item()
    rep["tests"].update(acc_original=base, acc_zero_all=zero, acc_restored=restored,
                        restored_max_abs_diff=d_rest, pristine_buffer=P.check_pristine())

    checks = {
        "identity_exact": d_id == 0.0,
        "patch_takes_effect": d_b0 > 1e-3,
        "zero_all_degrades": all(zero[t] <= base[t] - 0.3 or base[t] < 0.3 for t in a.tasks),
        "restore_exact": d_rest == 0.0 and base == restored and P.check_pristine(),
    }
    rep["checks"] = checks
    rep["gate_g3"] = all(checks.values())
    for k, v in rep["tests"].items():
        print(f"{k:28s} {v}")
    for k, v in checks.items():
        print(f"{k:28s} {'PASS' if v else 'FAIL'}")
    print("GATE G3:", "PASS" if rep["gate_g3"] else "FAIL")
    out = f"{cfg['results_dir']}/{model_slug(cfg['model_id'])}/g3_patch_check.json"
    write_json(out, rep)
    print(f"saved -> {out}")
    if not rep["gate_g3"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
