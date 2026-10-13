"""PyTorch Dataset for L2 speech CSV metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from .asr_backend import (
    is_qwen3_asr,
    qwen_assistant_target,
    qwen_language_name,
    to_qwen_audio_array,
)
from .audio_utils import load_and_preprocess_audio, parse_audio_path

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("audio_path", "text")


class L2SpeechDataset(Dataset):
    """Load utterances from a metadata CSV.

    Expected columns: ``audio_path``, ``text``, optional ``speaker_id``, ``split``.
    When ``text_column`` is set (e.g. ``pseudo_text``), that column is used as the
    training target instead of ``text``.
    """

    def __init__(
        self,
        csv_path: str,
        processor: Any,
        target_sr: int = 16000,
        text_column: str = "text",
        hypothesis_column: Optional[str] = None,
        max_audio_seconds: Optional[float] = 30.0,
        max_label_tokens: Optional[int] = None,
        skipped_rows_path: Optional[str] = None,
        language: str = "en",
    ) -> None:
        self.csv_path = str(csv_path)
        self.processor = processor
        self.target_sr = target_sr
        self.text_column = text_column
        self.hypothesis_column = hypothesis_column
        self.max_audio_seconds = max_audio_seconds
        self.max_label_tokens = max_label_tokens
        self.skipped_rows_path = skipped_rows_path
        self.language = language
        self.asr_family = "qwen3_asr" if is_qwen3_asr(processor) else "whisper"
        self.max_samples = (
            int(self.max_audio_seconds * self.target_sr) if self.max_audio_seconds is not None else None
        )

        df = pd.read_csv(self.csv_path)
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"{self.csv_path} missing required column '{col}'")
        if text_column not in df.columns:
            raise ValueError(f"{self.csv_path} missing text column '{text_column}'")
        if hypothesis_column is not None and hypothesis_column not in df.columns:
            raise ValueError(f"{self.csv_path} missing hypothesis column '{hypothesis_column}'")

        df = df.dropna(subset=["audio_path", text_column]).reset_index(drop=True)
        df["audio_path"] = df["audio_path"].astype(str)
        df[text_column] = df[text_column].astype(str)

        def _audio_exists(path: str) -> bool:
            return Path(parse_audio_path(path)[0]).is_file()

        missing = [p for p in df["audio_path"] if not _audio_exists(p)]
        if missing:
            logger.warning(
                "Dropping %d / %d rows with missing audio (example: %s)",
                len(missing),
                len(df),
                missing[0],
            )
            df = df[df["audio_path"].map(_audio_exists)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No valid rows in {self.csv_path}")

        valid_rows: List[Dict[str, Any]] = []
        valid_label_ids: List[List[int]] = []
        valid_hyp_ids: List[Optional[List[int]]] = []
        skipped_rows: List[Dict[str, Any]] = []
        for original_index, row in df.iterrows():
            row_dict = row.to_dict()
            text = str(row[text_column]).strip()
            try:
                tokenize_text = (
                    qwen_assistant_target(text, self.language) if self.asr_family == "qwen3_asr" else text
                )
                label_ids = self.processor.tokenizer(tokenize_text).input_ids
                label_length = len(label_ids)
                hyp_ids = None
                if hypothesis_column is not None:
                    hyp_text = str(row[hypothesis_column]).strip()
                    if self.asr_family == "qwen3_asr":
                        hyp_text = qwen_assistant_target(hyp_text, self.language)
                    hyp_ids = self.processor.tokenizer(hyp_text).input_ids if hyp_text else []
            except Exception as exc:
                skipped_rows.append(
                    {
                        **row_dict,
                        "skip_reason": "tokenize_error",
                        "skip_error": repr(exc),
                        "label_length": None,
                        "max_label_tokens": self.max_label_tokens,
                        "original_index": original_index,
                    }
                )
                continue

            hyp_length = len(hyp_ids) if hyp_ids is not None else 0
            too_long = self.max_label_tokens is not None and (
                label_length > self.max_label_tokens or hyp_length > self.max_label_tokens
            )
            if too_long:
                skipped_rows.append(
                    {
                        **row_dict,
                        "skip_reason": "label_too_long",
                        "skip_error": "",
                        "label_length": max(label_length, hyp_length),
                        "max_label_tokens": self.max_label_tokens,
                        "original_index": original_index,
                    }
                )
                continue

            valid_rows.append(row_dict)
            valid_label_ids.append(label_ids)
            valid_hyp_ids.append(hyp_ids if hyp_ids else None)

        if skipped_rows:
            logger.warning(
                "Skipping %d rows from %s (text=%s); see %s",
                len(skipped_rows),
                self.csv_path,
                text_column,
                self.skipped_rows_path or "warning logs",
            )
            for skipped in skipped_rows[:3]:
                logger.warning(
                    "Skipped row idx=%s reason=%s audio=%s label_length=%s",
                    skipped["original_index"],
                    skipped["skip_reason"],
                    skipped.get("audio_path", ""),
                    skipped.get("label_length"),
                )
            if self.skipped_rows_path:
                skip_path = Path(self.skipped_rows_path)
                skip_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(skipped_rows).to_csv(skip_path, index=False)

        df = pd.DataFrame(valid_rows).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"All rows were filtered out from {self.csv_path}")

        self.records = df.to_dict(orient="records")
        self.labels = [torch.tensor(ids, dtype=torch.long) for ids in valid_label_ids]
        self.hyp_labels = [
            torch.tensor(ids, dtype=torch.long) if ids is not None else None for ids in valid_hyp_ids
        ]
        logger.info(
            "Loaded %d utterances from %s (text=%s, hypothesis=%s)",
            len(self.records),
            self.csv_path,
            text_column,
            hypothesis_column,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        start_sec = row.get("start_sec")
        end_sec = row.get("end_sec")
        try:
            start_val = float(start_sec) if start_sec is not None and str(start_sec) not in ("", "nan") else None
        except (TypeError, ValueError):
            start_val = None
        try:
            end_val = float(end_sec) if end_sec is not None and str(end_sec) not in ("", "nan") else None
        except (TypeError, ValueError):
            end_val = None
        waveform = load_and_preprocess_audio(
            row["audio_path"],
            target_sr=self.target_sr,
            start_sec=start_val,
            end_sec=end_val,
        )

        if self.max_samples is not None and waveform.numel() > self.max_samples:
            waveform = waveform[: self.max_samples]

        text = str(row[self.text_column]).strip()
        labels = self.labels[idx]
        item: Dict[str, Any] = {
            "labels": labels,
            "audio_path": row["audio_path"],
            "text": text,
            "speaker_id": str(row.get("speaker_id") or row.get("global_speaker_id") or ""),
        }
        if self.asr_family == "qwen3_asr":
            item["audio"] = to_qwen_audio_array(waveform)
            return item

        features = self.processor.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )
        item["input_features"] = features.input_features.squeeze(0)
        hyp = self.hyp_labels[idx]
        if hyp is not None:
            item["hyp_labels"] = hyp
        return item


class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad Whisper input features and label sequences for Seq2SeqTrainer."""

    def __init__(self, processor: Any, decoder_start_token_id: int) -> None:
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Whisper expects the decoder BOS token to be stripped from labels when present.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        if features and "hyp_labels" in features[0]:
            hyp_features = [{"input_ids": f["hyp_labels"]} for f in features]
            hyp_batch = self.processor.tokenizer.pad(hyp_features, return_tensors="pt")
            hyp_labels = hyp_batch["input_ids"].masked_fill(hyp_batch.attention_mask.ne(1), -100)
            if (hyp_labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                hyp_labels = hyp_labels[:, 1:]
            batch["hyp_labels"] = hyp_labels
        return batch


class DataCollatorQwen3ASR:
    """Build Qwen3-ASR chat-template batches with ``output_labels=True``."""

    def __init__(self, processor: Any, language: str = "en") -> None:
        self.processor = processor
        self.language = language
        self.language_name = qwen_language_name(language)

    def __call__(self, features: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict[str, torch.Tensor]:
        conversations = []
        for item in features:
            transcript = str(item["text"]).strip()
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [{"type": "audio", "audio": to_qwen_audio_array(item["audio"])}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"language {self.language_name}<asr_text>{transcript}",
                            }
                        ],
                    },
                ]
            )
        try:
            batch = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                return_dict=True,
                processor_kwargs={"output_labels": True},
            )
        except TypeError:
            batch = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                return_dict=True,
                output_labels=True,
            )
        batch = dict(batch)
        keep = (
            "input_ids",
            "attention_mask",
            "input_features",
            "input_features_mask",
            "labels",
        )
        return {k: batch[k] for k in keep if k in batch}
