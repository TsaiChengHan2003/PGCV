"""Audio loading / resampling and Whisper text normalization."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional, Tuple, Union

import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# Unique training-row id: /path/to.flac#t=12.27,36.43 (seconds on the source file).
_AUDIO_SLICE_RE = re.compile(
    r"^(?P<path>.*)#t=(?P<start>[-+]?\d+\.?\d*),(?P<end>[-+]?\d+\.?\d*)$"
)


def parse_audio_path(audio_path: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Split ``path#t=start,end`` into a corpus path and optional slice times."""
    text = str(audio_path)
    match = _AUDIO_SLICE_RE.match(text)
    if not match:
        return text, None, None
    return match.group("path"), float(match.group("start")), float(match.group("end"))


def _slice_bounds(
    n_frames: int,
    sample_rate: int,
    start_sec: Optional[float],
    end_sec: Optional[float],
) -> Tuple[int, int]:
    """Frame window for an optional ``[start_sec, end_sec)`` slice."""
    if start_sec is None and end_sec is None:
        return 0, int(n_frames)
    start_i = int(round(float(start_sec or 0.0) * sample_rate))
    start_i = max(0, min(start_i, int(n_frames)))
    if end_sec is None:
        return start_i, int(n_frames)
    end_i = int(round(float(end_sec) * sample_rate))
    end_i = max(start_i, min(end_i, int(n_frames)))
    return start_i, end_i


def _read_waveform(
    path: str,
    start_sec: Optional[float],
    end_sec: Optional[float],
    target_sr: int,
):
    """Read a slice. On a FLAC seek failure, read the file and slice in memory.

    libsndfile can raise ``flac decoder lost sync`` when several dataloader
    workers seek into the same FLAC, especially a window that runs into the
    last frame. The corpus file is left untouched.
    """
    info = sf.info(path)
    src_sr = int(info.samplerate) if info.samplerate else int(target_sr)
    start_i, end_i = _slice_bounds(int(info.frames), src_sr, start_sec, end_sec)
    try:
        return sf.read(
            path,
            start=start_i,
            frames=end_i - start_i,
            always_2d=True,
            dtype="float32",
        )
    except sf.LibsndfileError:
        logger.warning(
            "FLAC seek failed for %s [%s, %s]; reading the whole file instead",
            path,
            start_sec,
            end_sec,
        )
        waveform_np, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        rate = int(sample_rate) if sample_rate else src_sr
        start_i, end_i = _slice_bounds(len(waveform_np), rate, start_sec, end_sec)
        return waveform_np[start_i:end_i], sample_rate


def load_and_preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> torch.Tensor:
    """Load audio as mono float32 waveform at ``target_sr`` Hz.

    Does **not** apply external VAD so learner pauses / hesitations are preserved.
    ``audio_path`` may be ``/path/file.flac#t=start,end``. Never writes the file.
    """
    path, parsed_start, parsed_end = parse_audio_path(audio_path)
    if start_sec is None:
        start_sec = parsed_start
    if end_sec is None:
        end_sec = parsed_end
    try:
        waveform_np, sample_rate = _read_waveform(path, start_sec, end_sec, target_sr)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load audio: {audio_path}") from exc

    waveform = torch.from_numpy(waveform_np).transpose(0, 1).contiguous()

    if waveform.ndim != 2:
        raise ValueError(f"Unexpected waveform shape {tuple(waveform.shape)} for {audio_path}")

    # Multi-channel -> mono by channel mean (preserve energy better than first-channel only).
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        try:
            import librosa
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "librosa is required for audio resampling. "
                "Install with: pip install librosa"
            ) from exc
        resampled = librosa.resample(
            waveform.squeeze(0).cpu().numpy(),
            orig_sr=int(sample_rate),
            target_sr=int(target_sr),
        )
        waveform = torch.from_numpy(resampled).unsqueeze(0)

    return waveform.squeeze(0).contiguous()


@lru_cache(maxsize=1)
def _get_normalizer():
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for BasicTextNormalizer. "
            "Install with: pip install transformers"
        ) from exc
    return BasicTextNormalizer()


def normalize_text(text: str) -> str:
    """Apply Whisper's official ``BasicTextNormalizer``."""
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    return _get_normalizer()(text)


def ensure_device(device: Union[str, torch.device]) -> torch.device:
    """Resolve CUDA when available; fall back to CPU with a warning."""
    if isinstance(device, torch.device):
        resolved = device
    else:
        resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return resolved
