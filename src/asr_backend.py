"""ASR family helpers: Whisper vs Qwen3-ASR (transformers-native -hf checkpoints)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .py310_re_lock import install_py310_re_lock

install_py310_re_lock()

logger = logging.getLogger(__name__)

QWEN3_ASR_MIN_TRANSFORMERS = (5, 13, 0)

# ISO / short codes used in this repo -> Qwen3-ASR canonical names.
_QWEN_LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
}


def infer_asr_family(model_name_or_processor: Any) -> str:
    if not isinstance(model_name_or_processor, str):
        if hasattr(model_name_or_processor, "apply_transcription_request"):
            return "qwen3_asr"
        return "whisper"
    name = str(model_name_or_processor).lower()
    if "qwen3-asr" in name or "qwen3_asr" in name:
        return "qwen3_asr"
    return "whisper"


def family_slug(model_name: str) -> str:
    """Directory slug under data/checkpoint/vector/results (e.g. whisper_base, qwen3_asr_0_6b)."""
    tail = model_name.split("/")[-1]
    lower = tail.lower()
    if lower.startswith("whisper-"):
        return f"whisper_{lower[len('whisper-'):].replace('-', '_')}"
    if lower.endswith("-hf"):
        lower = lower[: -len("-hf")]
    return lower.replace("-", "_").replace(".", "_")


def whisper_family_slug(model_name: str) -> str:
    """Back-compat alias used by older TEEMI runners."""
    return family_slug(model_name)


def qwen_language_name(language: str) -> str:
    text = str(language).strip()
    if not text:
        return "English"
    mapped = _QWEN_LANGUAGE_NAMES.get(text.lower())
    if mapped:
        return mapped
    return text


def qwen_assistant_target(transcript: str, language: str = "en") -> str:
    """Match the pretrained Qwen3-ASR output format: ``language English<asr_text>...``."""
    return f"language {qwen_language_name(language)}<asr_text>{transcript.strip()}"


def to_qwen_audio_array(waveform: Any) -> Any:
    """Convert a waveform to the 1D float32 array Qwen3-ASR's feature extractor expects.

    ``apply_transcription_request(audio=[...])`` batches a list of 1D ``(T,)`` clips.
    Passing ``(C, T)`` / ``(1, T)`` makes the extractor emit a 4D pad tensor and crash
    with ``ValueError: axes don't match array``.
    """
    import numpy as np

    if torch.is_tensor(waveform):
        array = waveform.detach().cpu().numpy()
    else:
        array = np.asarray(waveform)
    array = np.squeeze(array)
    if array.ndim == 2:
        # (C, T) or (T, C): keep the longer axis as time.
        array = array[0] if array.shape[0] <= array.shape[1] else array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"Qwen3-ASR audio must be 1D after squeeze, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def to_qwen_audio_batch(audios: Any) -> list:
    """Normalize a batch of waveforms for ``apply_transcription_request`` / chat audio."""
    if audios is None:
        return []
    if not isinstance(audios, (list, tuple)):
        return [to_qwen_audio_array(audios)]
    return [to_qwen_audio_array(item) for item in audios]


def _parse_transformers_version(raw: str) -> tuple:
    parts = []
    for chunk in raw.split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits) if digits else 0)
        if len(parts) >= 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def require_qwen3_asr_transformers() -> None:
    import os

    install_py310_re_lock()
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    import transformers

    version = _parse_transformers_version(transformers.__version__)
    if version < QWEN3_ASR_MIN_TRANSFORMERS:
        needed = ".".join(str(x) for x in QWEN3_ASR_MIN_TRANSFORMERS)
        raise ImportError(
            f"Qwen3-ASR native checkpoints need transformers>={needed} "
            f"(found {transformers.__version__} in {transformers.__file__}). "
            "Do not use eztalking_v2 / the Whisper env. "
            "Activate `l2-pseudo2real` (Python 3.10 + transformers 5.x) or set "
            "PYTHON_BIN to that environment's python."
        )


def is_qwen3_asr(model_name_or_processor: Any) -> bool:
    return infer_asr_family(model_name_or_processor) == "qwen3_asr"


def load_processor(model_name: str):
    if is_qwen3_asr(model_name):
        require_qwen3_asr_transformers()
        from transformers import AutoProcessor

        logger.info("Loading Qwen3-ASR processor %s", model_name)
        return AutoProcessor.from_pretrained(model_name)
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(model_name)


def _qwen_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _from_pretrained_qwen(model_name: str, **kwargs):
    require_qwen3_asr_transformers()
    try:
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)
    except (ImportError, ValueError, OSError):
        from transformers import Qwen3ASRForConditionalGeneration

        return Qwen3ASRForConditionalGeneration.from_pretrained(model_name, **kwargs)


def _has_full_weights(path: Path) -> bool:
    if (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file():
        return True
    return any(path.glob("model-*.safetensors")) or any(path.glob("pytorch_model-*.bin"))


def _load_adapter_only(model_name: str, device: Optional[torch.device]):
    """Merge a PEFT adapter dir onto its base model when full weights are absent."""
    path = Path(model_name)
    if not path.is_dir() or not (path / "adapter_config.json").is_file() or _has_full_weights(path):
        return None
    from peft import PeftConfig, PeftModel

    peft_cfg = PeftConfig.from_pretrained(str(path))
    base_id = peft_cfg.base_model_name_or_path
    logger.info("Loading LoRA adapter %s on base %s", path, base_id)
    base = load_asr_model(base_id, device=None)
    model = PeftModel.from_pretrained(base, str(path))
    model = model.merge_and_unload()
    if device is not None:
        model = model.to(device)
    return model


def load_asr_model(model_name: str, device: Optional[torch.device] = None):
    adapter = _load_adapter_only(model_name, device)
    if adapter is not None:
        return adapter
    if is_qwen3_asr(model_name):
        dtype = _qwen_torch_dtype()
        kwargs: Dict[str, Any] = {}
        try:
            model = _from_pretrained_qwen(model_name, dtype=dtype)
        except TypeError:
            model = _from_pretrained_qwen(model_name, torch_dtype=dtype)
        if device is not None:
            model = model.to(device)
        logger.info("Loaded Qwen3-ASR %s dtype=%s device=%s", model_name, dtype, device)
        return model

    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    if device is not None:
        model = model.to(device)
    return model


def load_pretrained_state_dict(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Load a state dict from a local HF dir or hub id (Whisper or Qwen3-ASR)."""
    path = Path(ckpt_dir)
    source = str(path) if path.exists() else ckpt_dir
    model = load_asr_model(source, device=torch.device("cpu"))
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    return state


def move_qwen_batch_to_device(
    batch: Any,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """Move processor outputs; keep integer tensors intact."""
    if hasattr(batch, "items"):
        items = dict(batch)
    else:
        items = dict(batch)
    out: Dict[str, Any] = {}
    for key, value in items.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        tensor = value.to(device)
        if dtype is not None and torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=dtype)
        out[key] = tensor
    return out


# Keys Qwen3-ASR.generate / forward actually read. Extra processor tensors are
# forwarded into the audio encoder and corrupt cu_seqlens window splits.
_QWEN_GENERATE_KEYS = {
    "input_ids",
    "attention_mask",
    "input_features",
    "input_features_mask",
    "position_ids",
    "inputs_embeds",
}


def qwen_model_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only tensors the Qwen3-ASR generate path consumes."""
    return {k: v for k, v in batch.items() if k in _QWEN_GENERATE_KEYS and torch.is_tensor(v)}


def decode_qwen_transcriptions(processor, generated_ids) -> list:
    try:
        texts = processor.decode(generated_ids, return_format="transcription_only")
    except TypeError:
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    if isinstance(texts, str):
        return [texts]
    return list(texts)
