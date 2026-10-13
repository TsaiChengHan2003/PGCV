"""Copy HuggingFace Whisper (+ optional τ) into an openai-whisper model.

Keeps the official long-form ``model.transcribe`` path used for the 13.7 SpWER
probe, while applying this repo's HF-space correction vectors.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_EXACT = {
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.layer_norm.bias": "decoder.ln.bias",
}

_REPLACEMENTS = (
    (".self_attn.q_proj.", ".attn.query."),
    (".self_attn.k_proj.", ".attn.key."),
    (".self_attn.v_proj.", ".attn.value."),
    (".self_attn.out_proj.", ".attn.out."),
    (".self_attn_layer_norm.", ".attn_ln."),
    (".encoder_attn.q_proj.", ".cross_attn.query."),
        (".encoder_attn.k_proj.", ".cross_attn.key."),
    (".encoder_attn.v_proj.", ".cross_attn.value."),
    (".encoder_attn.out_proj.", ".cross_attn.out."),
    (".encoder_attn_layer_norm.", ".cross_attn_ln."),
    (".final_layer_norm.", ".mlp_ln."),
    (".fc1.", ".mlp.0."),
    (".fc2.", ".mlp.2."),
)


def hf_key_to_openai(key: str) -> Optional[str]:
    name = key[6:] if key.startswith("model.") else key
    if name.startswith("proj_out"):
        return None
    if name in _EXACT:
        return _EXACT[name]
    name = name.replace(".layers.", ".blocks.")
    for src, dst in _REPLACEMENTS:
        name = name.replace(src, dst)
    return name


def hf_state_to_openai(hf_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped: Dict[str, torch.Tensor] = {}
    skipped = 0
    for key, tensor in hf_state.items():
        dest = hf_key_to_openai(key)
        if dest is None:
            skipped += 1
            continue
        mapped[dest] = tensor.detach().contiguous()
    logger.info("Mapped %d HF tensors to openai-whisper (%d skipped)", len(mapped), skipped)
    return mapped


def copy_hf_state_into_openai(openai_model: nn.Module, hf_state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """In-place copy of overlapping float weights. Returns (copied, skipped)."""
    mapped = hf_state_to_openai(hf_state)
    oa_sd = openai_model.state_dict()
    copied = 0
    skipped = 0
    missing = []
    for key, tensor in mapped.items():
        if key not in oa_sd:
            skipped += 1
            missing.append(key)
            continue
        dest = oa_sd[key]
        if dest.shape != tensor.shape:
            logger.warning("Skip %s: shape %s vs %s", key, tuple(dest.shape), tuple(tensor.shape))
            skipped += 1
            continue
        dest.copy_(tensor.to(device=dest.device, dtype=dest.dtype))
        copied += 1
    if missing:
        logger.info("HF keys with no openai counterpart (%d): %s", len(missing), missing[:8])
    logger.info("Copied %d tensors into openai-whisper (%d skipped)", copied, skipped)
    return copied, skipped


def load_openai_whisper_with_tau(
    *,
    openai_name: str = "base",
    student_model: str = "openai/whisper-base",
    tau_path: Optional[str] = None,
    lambda_val: float = 0.0,
    device: str = "cuda",
    model_dir: str = "",
):
    """openai-whisper model; λ=0 is vanilla ``openai_name``, else HF student + τ."""
    import whisper

    from .task_vector import apply_task_vector, load_tau

    model = whisper.load_model(openai_name, device=device, download_root=model_dir or None)
    if tau_path is None or abs(float(lambda_val)) < 1e-12:
        return model

    from .asr_backend import load_asr_model

    logger.info(
        "Loading HF %s and applying tau %s λ=%.3f",
        student_model,
        tau_path,
        float(lambda_val),
    )
    hf = load_asr_model(student_model, device=torch.device("cpu"))
    tau = load_tau(str(tau_path))
    apply_task_vector(hf, tau, float(lambda_val))
    copy_hf_state_into_openai(model, hf.state_dict())
    del hf
    model.eval()
    return model


def load_openai_whisper_from_hf(
    *,
    hf_model: str,
    openai_name: str = "base",
    device: str = "cuda",
    model_dir: str = "",
):
    """Load a HuggingFace Whisper checkpoint into openai-whisper ``transcribe``."""
    import whisper

    from .asr_backend import load_asr_model

    model = whisper.load_model(openai_name, device=device, download_root=model_dir or None)
    logger.info("Loading HF Whisper weights from %s", hf_model)
    hf = load_asr_model(hf_model, device=torch.device("cpu"))
    copy_hf_state_into_openai(model, hf.state_dict())
    del hf
    model.eval()
    return model
