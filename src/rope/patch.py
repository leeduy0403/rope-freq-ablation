"""Patch rotary inv_freq in place on a loaded model, with a pristine copy to restore.

Written against what the model actually exposes: every module that owns an
`inv_freq` buffer is patched. Run `python -m src.rope.patch --inspect` first.
"""
import torch

from src.rope.bands import band_slices

CACHE_ATTRS = ("_cos_cached", "_sin_cached", "cos_cached", "sin_cached")


def find_rotary_modules(model):
    found = []
    for name, mod in model.named_modules():
        if "inv_freq" in getattr(mod, "_buffers", {}) and mod._buffers["inv_freq"] is not None:
            found.append((name, mod))
    return found


def describe(model):
    lines = []
    for name, buf in model.named_buffers():
        if name.endswith("inv_freq"):
            lines.append(f"buffer {name}: shape={tuple(buf.shape)} dtype={buf.dtype} device={buf.device}")
    for name, mod in find_rotary_modules(model):
        extra = {a: getattr(mod, a) for a in ("rope_type", "attention_scaling") if hasattr(mod, a)}
        caches = [a for a in CACHE_ATTRS if hasattr(mod, a)]
        lines.append(f"module {name or '<root>'}: {type(mod).__name__} {extra} caches={caches}")
    return "\n".join(lines)


class RopePatcher:
    def __init__(self, model):
        self.mods = find_rotary_modules(model)
        assert self.mods, "no rotary module with inv_freq found — inspect the model structure"
        for name, mod in self.mods:
            rt = getattr(mod, "rope_type", "default")
            if rt != "default":
                raise RuntimeError(f"{name}: rope_type={rt!r}; patch assumes default RoPE")
        self.pristine = {name: mod.inv_freq.detach().clone() for name, mod in self.mods}
        self.state = "original"

    @property
    def inv_freq(self):
        """Pristine inv_freq (first rotary module) as float64 CPU tensor."""
        return next(iter(self.pristine.values())).double().cpu()

    def _set(self, name, mod, new):
        mod.inv_freq.copy_(new.to(mod.inv_freq.device, mod.inv_freq.dtype))
        for a in CACHE_ATTRS:
            if hasattr(mod, a) and not isinstance(getattr(mod, a), torch.nn.Parameter):
                if a in mod._buffers:
                    mod._buffers[a] = None
                else:
                    delattr(mod, a)
        if hasattr(mod, "_set_cos_sin_cache") and hasattr(mod, "max_seq_len_cached"):
            mod._set_cos_sin_cache(mod.max_seq_len_cached, mod.inv_freq.device, torch.float32)

    @torch.no_grad()
    def apply(self, transform, label="custom"):
        """transform: (inv_freq float32 tensor) -> tensor of same shape."""
        for name, mod in self.mods:
            orig = self.pristine[name].float()
            new = transform(orig.clone())
            assert new.shape == orig.shape
            self._set(name, mod, new)
        self.state = label
        return len(self.mods)

    @torch.no_grad()
    def restore(self):
        for name, mod in self.mods:
            self._set(name, mod, self.pristine[name])
        self.state = "original"

    def check_pristine(self):
        return all(torch.equal(mod.inv_freq, self.pristine[n]) for n, mod in self.mods)


def band_transform(head_dim, n_bands, band, mode="compress", factor=8.0):
    sl = band_slices(head_dim, n_bands)[band]

    def fn(inv):
        if mode == "compress":
            inv[sl] = inv[sl] / factor
        elif mode == "zero":
            inv[sl] = 0.0
        else:
            raise ValueError(mode)
        return inv

    return fn


def identity(inv):
    return inv


def zero_all(inv):
    return torch.zeros_like(inv)


if __name__ == "__main__":
    import argparse

    from src.config import load_config
    from src.model_io import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model-id")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    model, _ = load_model(load_config(args.config, {"model_id": args.model_id}))
    print(describe(model))
