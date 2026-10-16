# RoPE Frequency-Band Ablation — Implementation Guide

**Hand this file to Claude Code as the project spec.**
Course project IT4772 (NLP), Topic 12: *Long-Context LLMs — do LLMs really understand very long context?*

---

## 0. Instructions for the coding agent

Read this whole file before writing code.

**Work phase by phase. Each phase has an acceptance gate. Do not start phase N+1 until phase N's gate passes.** Report gate results to the user and stop if a gate fails — do not work around a failed gate silently.

**Hard rules:**

1. Never fabricate results. If a run fails, report the failure. An empty cell is acceptable; an invented number is not.
2. Every result cell is written to disk as JSON immediately after it is computed. The pipeline must be resumable: re-running must skip cells that already exist unless `--force` is passed.
3. Fix seeds everywhere. Use greedy decoding (`do_sample=False`). Record the seed, model revision, and library versions in every result file.
4. The library API for RoPE patching changes between `transformers` versions. **Inspect the installed version and the actual model class before writing the patch.** Do not copy the code sketches in this file verbatim — they show intent, not a guaranteed-working API.
5. Keep GPU cost low. Default to the smallest configuration that answers the question. Do not add extra lengths, bands, or examples without asking.

---

## 1. What we are measuring and why

### Background

RoPE rotates query/key vectors at position $m$ by angle $m\theta_i$, where

$$\theta_i = b^{-2i/d}, \qquad i = 0,\dots,d/2-1$$

with base $b$ (usually 10000) and head dimension $d$. Low $i$ = high frequency = short wavelength = local information. High $i$ = low frequency = long wavelength = global information.

Every context-extension method (PI, NTK-aware, YaRN) works by **compressing some of these frequencies** so that position angles at long range stay inside the distribution the model saw during training. They differ only in *which* frequencies they compress and by how much:

| Method | Effect on $\theta_i$ |
|---|---|
| PI | $\theta_i \to \theta_i / s$ for all $i$ (uniform) |
| NTK-aware | $b \to b\,s^{d/(d-2)}$: high freq barely touched, low freq compressed ≈ $1/s$ |
| YaRN | Explicit ramp: extrapolate high freq, fully interpolate low freq, blend in between |

### The question nobody has answered directly

**Which frequency bands does each task actually depend on — and do the extension methods compress exactly the bands the hard tasks need?**

If multi-hop tracing depends on bands that YaRN compresses heavily while needle-retrieval depends on bands YaRN preserves, that gives a mechanistic explanation for the result in RULER: near-perfect NIAH coexisting with collapse on tracing and aggregation.

### Hypothesis

> Lexical-match retrieval (NIAH) depends mainly on high-frequency bands, which extension methods preserve. Long-range semantic linking (variable tracking, aggregation) depends more on low-frequency bands, which extension methods compress hardest. Therefore the measured importance profile of a task should predict how badly that task degrades under extension.

**This produces a usable result whether the hypothesis holds or not.** If the profiles are flat or identical across tasks, that is evidence that frequency allocation is not the mechanism — also a finding.

### Why this design is cheap

- The intervention is a change to one buffer of numbers. No architecture surgery, no per-head tables, no GQA constraint — it applies uniformly to all heads.
- Half the contribution (the compression profile of each method) is **pure arithmetic with no GPU at all**.
- Small models are fine. Short contexts are fine.

---

## 2. Deliverables

| # | Artifact | Produced by |
|---|---|---|
| D1 | **Importance heatmap**: band × task × length — accuracy drop when each band is suppressed | Phase 3 |
| D2 | **Compression profile**: band × method — how much PI / NTK / YaRN compress each band | Phase 4 (CPU only) |
| D3 | **Overlay figure**: D1 and D2 on the same band axis, per task | Phase 5 |
| D4 | **Correlation table**: per-task alignment between band importance and method compression | Phase 5 |
| D5 | **Baseline table**: unmodified model accuracy, all tasks × lengths, with CIs | Phase 2 |
| D6 | Results JSON, plotting scripts, README, fixed seeds | All phases |

D3 is the money figure for the presentation slide.

---

## 3. Repository layout

```
rope-freq-ablation/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml           # model id, lengths, bands, n_examples, seed
├── src/
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── base.py            # Task interface: generate(), score()
│   │   ├── niah.py            # S-NIAH and MK-NIAH
│   │   ├── vt.py              # Variable Tracking
│   │   └── fwe.py             # Frequent Words Extraction
│   ├── rope/
│   │   ├── bands.py           # band partitioning, band masks
│   │   ├── patch.py           # apply a modified inv_freq to a loaded model
│   │   └── methods.py         # PI / NTK-aware / YaRN theta profiles (CPU only)
│   ├── eval/
│   │   ├── runner.py          # cell loop, caching, resume
│   │   └── metrics.py         # accuracy + bootstrap CI
│   └── analyze/
│       ├── heatmap.py
│       └── overlay.py
├── results/                   # one JSON per cell; committed
└── figures/
```

---

## 4. Configuration defaults

```yaml
model_id: null                 # MUST be chosen in Phase 1, do not hardcode a guess
dtype: bfloat16
load_in_4bit: false            # enable only if VRAM-bound

lengths: [1024, 2048, 4096]    # start here; extend only after Phase 2 passes
n_examples: 50                 # per cell; raise to 100 if time allows
n_bands: 8
seed: 1234

tasks: [s_niah, mk_niah, vt, fwe]

intervention:
  mode: compress               # "compress" (primary) or "zero" (extreme control)
  compress_factor: 8.0         # theta_i -> theta_i / factor, for the targeted band only
```

**Do not use `lengths` beyond the model's native context window.** This experiment is about which frequencies matter *within* the trained range. Extension behaviour is handled analytically in Phase 4.

---

## 5. Phases

### Phase 0 — Environment

Install `torch`, `transformers`, `accelerate`, `numpy`, `pandas`, `matplotlib`, `pyyaml`. Optionally `bitsandbytes` for 4-bit.

Write `src/env_report.py` that prints: Python version, torch version, transformers version, CUDA availability, GPU name, total and free VRAM. Save the output to `results/env.json`.

**Gate G0:** `env_report.py` runs and a GPU is visible. If no GPU, stop and report — this project needs one, though a free-tier T4 is sufficient.

---

### Phase 1 — Model selection

Candidate requirements, in priority order:

1. Uses RoPE (config has `rope_theta` or an equivalent rotary attribute).
2. Instruction-tuned (base models are unreliable at following the task format).
3. 0.5B–1.5B parameters.
4. Native context ≥ 4096.

GQA is **fine** for this experiment — the intervention is uniform across heads, so the relative-position property is preserved. This is the main reason this design is cheaper than the per-head alternative.

Write `src/select_model.py` that takes a list of candidate model ids, loads each config, and prints a table:

```python
from transformers import AutoConfig

def inspect(model_id):
    c = AutoConfig.from_pretrained(model_id)
    return {
        "model_id": model_id,
        "layers": c.num_hidden_layers,
        "q_heads": c.num_attention_heads,
        "kv_heads": getattr(c, "num_key_value_heads", c.num_attention_heads),
        "head_dim": getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
        "rope_theta": getattr(c, "rope_theta", None),
        "max_pos": c.max_position_embeddings,
        "rope_scaling": getattr(c, "rope_scaling", None),
    }
```

**Ask the user to supply the candidate list.** Do not guess model ids — hallucinated repo names waste time. Report the table and let the user pick.

**Gate G1:** a model is chosen, loads successfully, and produces coherent output on a trivial prompt.

---

### Phase 2 — Tasks and baseline (the most important gate)

#### Task generators

All four tasks are generated in pure Python. No dataset downloads required.

**Haystack filler.** Ask the user whether to use a real text corpus or synthetic filler. Synthetic filler (a pool of ~50 neutral sentences, sampled with a fixed seed) is acceptable and keeps the project dependency-free. Record the choice in the results metadata.

**`s_niah`** — insert one sentence of the form `The special magic number for <KEY> is <7-digit number>.` at a given depth. Question asks for the number. Score: exact-match of the number anywhere in the output.

**`mk_niah`** — same, plus 3 distractor needles with different keys. Score the target key's value only.

**`vt`** — emit a chain `X1 = 54321`, then `X2 = X1`, `X3 = X2`, `X4 = X3`, scattered through the haystack in random order. Question: list all variables equal to 54321. Score: exact set match (all four names, no extras).

**`fwe`** — generate words drawn from a Zeta distribution (exponent 2.0), so a few words are far more frequent. Question: the three most frequent words. Score: exact set match of the top 3.

Every generator takes `(length_tokens, seed, depth_fraction)` and returns `(prompt, gold_answer)`. Lengths must be measured with the **model's own tokenizer**, not by character count — pad or trim the filler until the tokenized prompt is within ±2% of the target.

Append a short answer prefix to each prompt (e.g. `Answer:`) so the model does not refuse or ramble. Use the model's chat template if it has one.

#### Baseline run

Evaluate the **unmodified** model on all tasks × lengths. Report accuracy with a bootstrap 95% CI (1000 resamples).

**Gate G2 (hard gate — the project depends on this):**

> At the **shortest** length, the model must score **above 60%** on `s_niah` and **above 30%** on `vt` and `fwe`.

If the model cannot do a task at short length, that task carries no signal and ablating frequencies on it measures nothing. Remedies, in order:

1. Simplify: fewer hops in `vt` (try 2 instead of 4), fewer distractors in `mk_niah`, stronger frequency skew in `fwe`.
2. Improve the prompt: add one in-context example.
3. Try the next candidate model.
4. If `vt` and `fwe` both fail on every candidate, **stop and report to the user**. The experiment needs at least one hard task to contrast against `s_niah`.

Do not proceed past a failed G2. This is the single most common way a project like this collapses in week four.

---

### Phase 3 — Frequency-band ablation (the GPU work)

#### Band partitioning

With head dimension $d$, there are $d/2$ frequency pairs. Split them into `n_bands` contiguous groups, band 0 = highest frequency.

```python
def band_slices(head_dim, n_bands):
    n_freq = head_dim // 2
    edges = [round(k * n_freq / n_bands) for k in range(n_bands + 1)]
    return [slice(edges[k], edges[k+1]) for k in range(n_bands)]
```

Also record, for each band, the **wavelength range** $\lambda = 2\pi/\theta$ in tokens. This is what makes the heatmap interpretable: a band whose wavelengths are 10–40 tokens is local; one with wavelengths of 10k tokens is global.

#### The intervention

For the targeted band only:

- `compress` mode: $\theta_i \to \theta_i / f$ with $f$ = `compress_factor`. This mimics what interpolation actually does — it blurs that band's positional resolution without removing it.
- `zero` mode: $\theta_i \to 0$. The band carries no position information at all. Use as an extreme control on a couple of bands, not as the main sweep.

Prefer `compress`. `zero` is informative but may break the model so thoroughly that every band looks equally critical.

#### Patching

In current Llama-style models, rotary frequencies live in an `inv_freq` buffer on a rotary-embedding module. Where that module sits (per attention layer vs once on the model) **has changed across `transformers` releases**.

Procedure for the agent:

1. Load the model.
2. Walk `model.named_buffers()` and `model.named_modules()` to find every tensor named `inv_freq` and every module exposing one. Print what you find.
3. Write the patch against what actually exists, not against a remembered API.
4. Clear any cached cos/sin tables the module holds (attributes with names like `_cos_cached` / `_sin_cached`), if present.
5. Keep a pristine copy of the original `inv_freq` and restore it between configurations — **do not reload the model for every cell**, that dominates runtime.

Sketch of intent:

```python
def patch_inv_freq(model, transform):
    """transform: (inv_freq: Tensor) -> Tensor, same shape and dtype."""
    patched = 0
    for module in model.modules():
        if hasattr(module, "inv_freq") and module.inv_freq is not None:
            new = transform(module.original_inv_freq_backup)
            module.inv_freq = new.to(module.inv_freq.device, module.inv_freq.dtype)
            for attr in ("_cos_cached", "_sin_cached"):
                if hasattr(module, attr):
                    delattr(module, attr)
            patched += 1
    assert patched > 0, "no rotary module found — inspect the model structure"
    return patched
```

**Gate G3 — the patch must be proved correct before any sweep:**

| Test | Expectation |
|---|---|
| Identity transform, then compare logits on a fixed prompt against an unpatched copy | Identical to floating-point tolerance |
| Zero **all** frequencies | Large degradation on every task (model becomes position-blind) |
| Restore original, re-run baseline | Matches Phase 2 numbers exactly |

If the identity test does not pass, nothing downstream means anything. Report and stop.

#### Sweep

```
for length in lengths:
  for task in tasks:
    for band in range(n_bands):
      run n_examples, write results/{task}_{length}_band{band}.json
```

Plus one `band = none` cell per (task, length) as the in-sweep baseline.

Cost: `n_bands + 1` × `n_tasks` × `n_lengths` × `n_examples` generations. With the defaults that is 9 × 4 × 3 × 50 = **5,400 generations** — single-digit hours on a free-tier T4 with a 1B model.

Each JSON must contain: config hash, model id and revision, library versions, seed, band index, wavelength range, raw per-example correctness, accuracy, bootstrap CI, wall-clock time.

**Gate G4:** the `band = none` cells reproduce the Phase 2 baseline. If they do not, there is state leaking between configurations — most likely a stale cos/sin cache.

---

### Phase 4 — Compression profiles (no GPU)

Pure arithmetic. For each extension method and each scale factor $s \in \{2, 4, 8\}$, compute the per-dimension compression ratio

$$c_i = \frac{\theta_i^{\text{method}}}{\theta_i^{\text{original}}}$$

then average within each band.

```python
def theta_original(head_dim, base):
    i = np.arange(0, head_dim, 2)[: head_dim // 2]
    return base ** (-i / head_dim)

def theta_pi(theta, s):
    return theta / s

def theta_ntk(head_dim, base, s):
    return theta_original(head_dim, base * s ** (head_dim / (head_dim - 2)))

def theta_yarn(theta, s, L_orig, alpha=1.0, beta=32.0):
    wavelen = 2 * np.pi / theta
    r = L_orig / wavelen                    # rotations completed within original context
    gamma = np.clip((beta - r) / (beta - alpha), 0.0, 1.0)
    return (1 - gamma) * theta + gamma * (theta / s)
```

Check the YaRN ramp direction against the paper before trusting it: high-frequency dimensions (large $r$) must end up with $\gamma = 0$, i.e. **not** interpolated. Write a unit test asserting $\gamma \to 0$ for the highest-frequency dimension and $\gamma \to 1$ for the lowest.

Output: a `n_bands × n_methods × n_scales` table, D2.

---

### Phase 5 — Analysis

**D1 heatmap.** Band (x) × task (y), cell = accuracy drop versus the `band = none` baseline, one panel per length. Annotate the x-axis with each band's wavelength range in tokens, not just the band index.

**D3 overlay.** For each task, plot band importance (from D1) and band compression (from D2, per method) on the same band axis. This is the slide.

**D4 correlation.** For each (task, method, scale), compute the correlation between the band importance vector and the band compression vector. A high correlation means *the method compresses exactly the bands this task needs*.

**Write the interpretation honestly.** Three possible outcomes, all reportable:

| Outcome | Reading |
|---|---|
| `s_niah` importance concentrated in high bands; `vt`/`fwe` in low bands | Hypothesis supported — mechanistic explanation for the RULER gap |
| Profiles broadly similar across tasks | Hypothesis not supported — frequency allocation is not what separates these tasks. Still a result |
| Importance flat across all bands | Intervention too weak or too strong. Re-run with a different `compress_factor` before concluding anything |

The third outcome is a methodology problem, not a finding. Check it before writing conclusions: sweep `compress_factor` ∈ {2, 8, 32} on a single task and length first, and pick the factor that produces a readable dynamic range.

---

## 6. Pitfalls

1. **Measuring length in characters instead of tokens.** Every length must be verified with the model's tokenizer.
2. **Stale cos/sin caches.** The most likely source of a silently wrong result. Gate G4 exists to catch it.
3. **Reloading the model per cell.** Patch in place; keep a pristine `inv_freq` copy.
4. **Lenient scoring on `vt`/`fwe`.** Require exact set match. Substring matching inflates these tasks badly.
5. **Needle depth confound.** Keep depth fixed (e.g. 50%) across the whole sweep, or sweep it separately — do not let it vary with band.
6. **Reporting a single averaged score.** The entire point is that tasks differ. Never average across tasks.
7. **No error bars.** Bootstrap CIs on every cell. This is the first thing a reviewer asks about.
8. **Running out of session time on free tiers.** Kaggle caps sessions at 12 hours with no persistent state. Write every cell to disk immediately and make the runner resumable.

---

## 7. Minimum viable configuration

If time or compute runs short, cut in this order:

1. Lengths: 3 → 2
2. `n_examples`: 50 → 30 (and widen the reported CIs accordingly, do not hide this)
3. `n_bands`: 8 → 4
4. Tasks: drop `mk_niah` (keep `s_niah`, `vt`, `fwe` — they span the contrast)

**Never cut:** the Phase 2 baseline, the `band = none` cells, or gate G3.

---

## 8. Final checklist

- [ ] `results/env.json` with full version info
- [ ] Model selection table, with the chosen model and the reason
- [ ] Baseline table (D5) with bootstrap CIs
- [ ] G3 patch-correctness evidence recorded in the repo
- [ ] Full band sweep JSONs, resumable, committed
- [ ] `compress_factor` calibration run documented
- [ ] D1 heatmap, D2 table, D3 overlay, D4 correlations
- [ ] README with exact commands to reproduce every figure
- [ ] Honest write-up stating which of the three outcomes occurred

---

## 9. Reference reading (for the report, not needed to run the code)

- *RULER: What's the Real Context Size of Your Long-Context Language Models?* — arXiv 2404.06654 (COLM 2024). Source of the four task categories.
- *YaRN: Efficient Context Window Extension of Large Language Models* — arXiv 2309.00071 (ICLR 2024). Source of the NTK-by-parts ramp.
- *Lost in the Middle* — Liu et al., TACL 2024, 12:157–173.
- *Same Task, More Tokens* (FLenQA) — arXiv 2402.14848 (ACL 2024). Why perplexity is not the metric.
- *Retrieval Head Mechanistically Explains Long-Context Factuality* — arXiv 2404.15574 (ICLR 2025). The head-level analogue of this frequency-level analysis.
- *A Controlled Study on Long Context Extension and Generalization in LLMs* — arXiv 2409.12181. Read before claiming novelty.
