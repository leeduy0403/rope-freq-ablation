"""Phase 1: inspect candidate model configs (no weights downloaded)."""
import argparse

import pandas as pd
from transformers import AutoConfig


def rope_info(c):
    # rope_theta moved into rope_parameters / rope_scaling in newer transformers
    theta = getattr(c, "rope_theta", None)
    params = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None)
    if theta is None and isinstance(params, dict):
        theta = params.get("rope_theta")
    return theta, params


def inspect(model_id):
    c = AutoConfig.from_pretrained(model_id)
    c = getattr(c, "text_config", c)
    theta, params = rope_info(c)
    return {
        "model_id": model_id,
        "layers": c.num_hidden_layers,
        "q_heads": c.num_attention_heads,
        "kv_heads": getattr(c, "num_key_value_heads", c.num_attention_heads),
        "head_dim": getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
        "rope_theta": theta,
        "max_pos": c.max_position_embeddings,
        "rope_params": params,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_ids", nargs="+")
    args = ap.parse_args()
    rows = []
    for mid in args.model_ids:
        try:
            rows.append(inspect(mid))
        except Exception as e:  # report, do not guess
            rows.append({"model_id": mid, "error": f"{type(e).__name__}: {e}"})
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
