"""Config loading, hashing and run metadata shared by every result file."""
import hashlib
import json
import platform
from pathlib import Path

import yaml


def load_config(path="config/default.yaml", overrides=None, sets=None):
    """overrides: {key: value} (None = keep). sets: ["a.b.c=yaml_value", ...] dotted overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    for item in sets or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def config_hash(cfg, keys=None):
    """Stable short hash of the config (or a subset of keys)."""
    sub = {k: cfg[k] for k in keys} if keys else cfg
    blob = json.dumps(sub, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def model_slug(model_id):
    return model_id.replace("/", "__")


def lib_versions():
    import numpy
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "cuda": torch.version.cuda,
    }


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)  # atomic: a killed session never leaves a half-written cell
