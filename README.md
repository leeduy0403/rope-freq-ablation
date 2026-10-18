# RoPE Frequency-Band Ablation

IT4772 (NLP), Topic 12 — *Long-context LLMs: which RoPE frequency bands does each task depend on, and do context-extension methods (PI / NTK / YaRN) compress exactly those bands?*

Method: for each band of rotary frequencies, compress `theta_i -> theta_i / f` **in place** on a loaded model (one `inv_freq` buffer, all heads), measure the accuracy drop on RULER-style synthetic tasks within the native context, then overlay with the analytical compression profile of each extension method.

Spec: `docs/rope-frequency-ablation-guide.md`.

## Setup

```bash
uv venv --python 3.10 .venv            # or python -m venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
pytest -q                               # CPU-only unit tests (tiny random Llama, no download)
```

## Running on Kaggle

Open `notebooks/kaggle_run.ipynb` on Kaggle (*File → Import Notebook*), set **GPU T4 x2** and **Internet On**, then run cells top to bottom — or *Run All*: every gate exits non-zero on failure and the notebook stops there. `compress_factor` is chosen automatically in the calibration cell (largest per-band accuracy spread; recorded in `results/<model>/calibration.json`). Long phases: *Save Version → Save & Run All* (background, ≤12 h). To resume, attach the previous version's output as input — cell 2 restores `results/` and finished cells are skipped. T4 has no native bf16, so `dtype: auto` picks fp16.

## Pipeline (phase by phase — do not skip a failed gate)

| Phase | Command | Output | Gate |
|---|---|---|---|
| 0 env | `python -m src.env_report` | `results/env.json` | G0: GPU visible |
| 1 model | `python -m src.select_model <id1> <id2> ...` | table on stdout | G1: chosen model loads & answers |
| | `python -m src.rope.patch --inspect` | rotary modules found | |
| 2 baseline | `python -m src.eval.runner --stage baseline` | `results/<model>/baseline/*.json`, `baseline_table.csv` | **G2**: shortest length s_niah>0.6, vt>0.3, fwe>0.3 |
| 3a patch check | `python -m src.check_patch --tasks s_niah vt` | `results/<model>/g3_patch_check.json` | **G3**: identity exact, patch takes effect, zero-all degrades, restore exact |
| 3b calibrate | `python -m src.eval.runner --stage sweep --tasks vt --lengths 2048 --factor 2` (then 8, 32) | `results/<model>/sweep/compress_f{2,8,32}/` | pick factor with readable dynamic range |
| 3c sweep | `python -m src.eval.runner --stage sweep` | `results/<model>/sweep/compress_f8/*.json` | **G4**: band=none == baseline |
| 4 profiles (CPU) | `python -m src.rope.methods --head-dim 64 --base 1000000 --l-orig 32768` | `results/compression_profile.csv` | unit tests pass |
| 5 analysis | `python -m src.analyze.heatmap` / `python -m src.analyze.overlay --scale 4` / `python -m src.analyze.shift` | `figures/*D1*`, `figures/*D3*`, `results/<model>/D2_*.csv`, `D4_*.csv` | |

Common flags: `--model-id`, `--tasks`, `--lengths`, `--n`, `--bands`, `--mode {compress,zero}`, `--factor`, `--force`.

Every cell is written to its own JSON as soon as it finishes (atomic write). Re-running skips existing cells unless `--force` — safe for Kaggle/Colab session limits.

Smoke test (few minutes, small GPU):

```bash
python -m src.eval.runner --stage baseline --tasks s_niah vt fwe --lengths 1024 --n 5 --results-dir results/_smoke
python -m src.eval.runner --stage sweep    --tasks s_niah --lengths 1024 --n 5 --bands 0 7 --results-dir results/_smoke
```

## Design notes

- **Tasks** (`src/tasks/`): `s_niah`, `mk_niah` (3 distractors), `vt` (4-hop chain, statements scattered in random order), `fwe` (Zeta α=3 coded words, top-3; α=2 was too hard for the 1.5B model, see the fwe probe cell). Pure Python, synthetic filler (50 neutral sentences, fixed seed). Prompt length is measured with the model's own tokenizer *after* the chat template, ±2%.
- **Scoring**: niah = exact number anywhere in output; vt / fwe = exact set match (no extras).
- **Same examples for every band**: example `k` uses seed `seed*1000+k`, independent of band → per-band differences are paired.
- **Needle depth** fixed at 0.5 for the target needle.
- **Patch** (`src/rope/patch.py`): finds every module owning an `inv_freq` buffer, keeps a pristine clone, writes transformed values in place, clears cos/sin caches if the module has them, restores between cells. Refuses non-default `rope_type`.
- **Bands**: rotary dim / 2 frequency pairs split into `n_bands` contiguous groups, band 0 = highest frequency; each result records the band's wavelength range `2π/θ` in tokens.
- **Compression** reported in bits `log2(θ_orig/θ_method)` (0 = untouched, log2 s = full interpolation). YaRN temperature term ignored (does not change θ). PI is uniform → correlation undefined (NaN), by construction.
- `dtype: auto` picks bf16 only on GPUs with native support (Ampere+); T4 / Pascal use fp16.

## Status

Run 1 (Kaggle T4, Qwen2.5-1.5B-Instruct, fp16, lengths 1024/2048/4096, n=50, compress factor 32): all gates G0–G4 pass. Results in `results/Qwen__Qwen2.5-1.5B-Instruct/`, figures in `figures/`.

- Band 0 (λ 6–28 tokens): compressing it breaks every task (degenerate output) — model-wide, reported separately.
- s_niah / mk_niah: no significant dependence on bands 1–7.
- vt: significant dependence on bands 1–4; fwe: bands 3–4. Band 4 (λ 6k–28k) only matters at 4096 → a band starts to matter once its wavelength approaches the context length.
- D4 correlations with NTK/YaRN compression are negative (methods compress bands 5–7, which no task uses within ≤4k).

Run 2a (`notebooks/kaggle_long.ipynb`, 8192, s_niah/vt/fwe, n=50): G4 passes. vt's band-4 drop grows with length — −0.02 / +0.04 / +0.18 / +0.34 at 1k/2k/4k/8k (relative to baseline: 0 / 6% / 26% / 59%) — while s_niah stays flat; band 5 (λ ≥ 35k) is still ≈0 at 8k. See `figures/*D7_shift*`.

Run 2b (16384): reduced design (vt + s_niah, bands 3–7, n≈30 after a timing probe) — a full run would take ~19 h on T4.

Analysis notes: drop CIs are paired bootstrap (same examples in every band); D4 is reported with and without band 0; Spearman is identical across scales for NTK/YaRN by construction; PI is uniform → NaN.
