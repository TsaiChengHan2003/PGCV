"""Teacher pseudo-label generation with greedy decoding."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .audio_utils import ensure_device, load_and_preprocess_audio
from .progress import progress_bar

logger = logging.getLogger(__name__)


class _PseudoLabelDataset(Dataset):
    def __init__(self, paths: List[str], target_sr: int, max_audio_seconds: float) -> None:
        self.paths = paths
        self.target_sr = target_sr
        self.max_samples = int(max_audio_seconds * target_sr)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.paths[idx]
        try:
            waveform = load_and_preprocess_audio(path, target_sr=self.target_sr)
            if waveform.numel() > self.max_samples:
                waveform = waveform[: self.max_samples]
            error = ""
        except Exception as exc:  # noqa: BLE001
            waveform = None
            error = repr(exc)
        return {
            "index": idx,
            "audio_path": path,
            "waveform": waveform,
            "error": error,
        }


@torch.inference_mode()
def generate_pseudo_labels(
    model_name: str,
    input_csv: str,
    output_csv: str,
    batch_size: int = 16,
    device: str = "cuda",
    language: str = "en",
    task: str = "transcribe",
    max_audio_seconds: float = 30.0,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Generate greedy pseudo-transcriptions and write a new CSV.

    Adds / overwrites column ``pseudo_text`` while preserving original columns.
    """
    device_t = ensure_device(device)
    logger.info("Loading teacher model %s on %s", model_name, device_t)

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device_t)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.forced_decoder_ids = forced_decoder_ids
    # Deterministic greedy decoding
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.temperature = 0.0

    df = pd.read_csv(input_csv)
    if "audio_path" not in df.columns:
        raise ValueError(f"{input_csv} must contain audio_path")

    paths: List[str] = df["audio_path"].astype(str).tolist()
    hypotheses: List[Optional[str]] = [None] * len(paths)
    target_sr = 16000
    dataset = _PseudoLabelDataset(paths, target_sr=target_sr, max_audio_seconds=max_audio_seconds)

    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        skipped = [
            (int(item["index"]), str(item["audio_path"]), str(item["error"]))
            for item in batch
            if item["waveform"] is None
        ]
        return {
            "indices": [int(item["index"]) for item in batch if item["waveform"] is not None],
            "audio_paths": [str(item["audio_path"]) for item in batch if item["waveform"] is not None],
            "waveforms": [item["waveform"] for item in batch if item["waveform"] is not None],
            "skipped": skipped,
        }

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    dataloader = DataLoader(**loader_kwargs)

    for batch in progress_bar(
        dataloader,
        total=len(dataloader),
        desc=f"pseudo-label {Path(input_csv).name}",
        unit="batch",
    ):
        for idx, path, error in batch["skipped"]:
            logger.error("Skip %s: %s", path, error)
            hypotheses[idx] = ""

        if not batch["waveforms"]:
            continue

        waveforms = [wav.numpy() for wav in batch["waveforms"]]
        inputs = processor.feature_extractor(
            waveforms,
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device_t)
        pred_ids = model.generate(
            input_features,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
        )
        texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
        for idx, text in zip(batch["indices"], texts):
            hypotheses[idx] = text.strip()

    df = df.copy()
    df["pseudo_text"] = [h if h is not None else "" for h in hypotheses]

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote pseudo labels to %s (%d rows)", out_path, len(df))
    return df
