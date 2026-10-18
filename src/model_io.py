"""Model loading and greedy generation."""
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name):
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, name)


def configure_attention(cfg):
    """Keep SDPA on the memory-efficient kernel.

    With no attention mask, transformers passes enable_gqa=True to SDPA. On GPUs without
    FlashAttention (T4 = sm75) the memory-efficient kernel does not support GQA, so PyTorch
    silently falls back to the math kernel, which materialises the full fp32 L x L score
    matrix (OOM at 16k, quadratic slowdown at 8k). Forcing transformers to repeat K/V lets the
    memory-efficient kernel run; disabling the math backend turns any further fallback into an
    error instead of a silent slowdown. Returns the setting recorded in result metadata.
    """
    mode = cfg.get("sdpa_kernel", "efficient")
    if mode == "efficient" and torch.cuda.is_available():
        import transformers.integrations.sdpa_attention as sdpa

        sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
        torch.backends.cuda.enable_math_sdp(False)
        return "sdpa-efficient(repeat_kv)"
    return "sdpa-default"


def load_model(cfg):
    dtype = resolve_dtype(cfg.get("dtype", "auto"))
    cfg["_attention"] = configure_attention(cfg)
    kw = dict(revision=cfg.get("revision", "main"),
              attn_implementation=cfg.get("attn_implementation", "sdpa"))
    # transformers renamed torch_dtype -> dtype in 4.56
    new_api = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)
    kw["dtype" if new_api else "torch_dtype"] = dtype
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = "auto"
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=kw["revision"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kw)
    if not cfg.get("load_in_4bit"):
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tok


def model_revision(model):
    return getattr(model.config, "_commit_hash", None)


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens):
    """Greedy decode. `prompt` is already chat-templated text."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
