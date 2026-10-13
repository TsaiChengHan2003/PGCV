"""Detailed WER / Ins / Sub / Del evaluation with CSV logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jiwer
import pandas as pd
import torch

from .asr_backend import (
    decode_qwen_transcriptions,
    is_qwen3_asr,
    move_qwen_batch_to_device,
    qwen_model_inputs,
    to_qwen_audio_batch,
)
from .audio_utils import normalize_text
from .progress import progress_bar

logger = logging.getLogger(__name__)


def _alignment_counts(reference: str, hypothesis: str) -> Tuple[int, int, int, int]:
    """Return (hits, substitutions, deletions, insertions) via jiwer process_words."""
    if not reference.strip() and not hypothesis.strip():
        return 0, 0, 0, 0
    out = jiwer.process_words(reference, hypothesis)
    return int(out.hits), int(out.substitutions), int(out.deletions), int(out.insertions)


def _slice_eval_batch(batch: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Take one utterance out of a collated eval batch."""
    sliced: Dict[str, Any] = {}
    width = None
    for key, value in batch.items():
        if isinstance(value, (list, tuple)):
            width = len(value) if width is None else width
            sliced[key] = [value[index]]
        elif torch.is_tensor(value) and value.dim() > 0 and (width is None or value.shape[0] == width):
            width = int(value.shape[0]) if width is None else width
            sliced[key] = value[index : index + 1]
        else:
            sliced[key] = value
    return sliced


def _as_str_list(values: Any, size: int) -> List[str]:
    if isinstance(values, str):
        return [values]
    if values is None:
        return [""] * size
    return [str(v) for v in values]


def _decode_batch(
    model: torch.nn.Module,
    processor,
    batch: Dict[str, Any],
    device: torch.device,
    gen_kwargs: Dict[str, Any],
    *,
    qwen: bool,
    with_scores: bool,
) -> Tuple[Sequence[str], Sequence[str], Sequence[str], Optional[Any], Optional[torch.Tensor]]:
    """Return refs, paths, hyps, and optionally generate outputs + generated token ids."""
    if qwen:
        if "audio" not in batch:
            raise KeyError("Qwen3-ASR eval batch must contain 'audio'")
        audio = to_qwen_audio_batch(batch["audio"])
        refs_raw = _as_str_list(batch.get("text"), len(audio))
        paths = _as_str_list(batch.get("audio_path"), len(audio))
        language = batch.get("language", "en")
        inputs = processor.apply_transcription_request(audio=audio, language=language)
        inputs = move_qwen_batch_to_device(inputs, device, dtype=getattr(model, "dtype", None))
        model_inputs = qwen_model_inputs(inputs)
        prompt_len = model_inputs["input_ids"].shape[1]
        outputs = model.generate(**model_inputs, **gen_kwargs)
        if with_scores:
            sequences = outputs.sequences
            generated = sequences[:, prompt_len:]
            hyps_raw = decode_qwen_transcriptions(processor, generated)
            return refs_raw, paths, hyps_raw, outputs, generated
        generated = outputs[:, prompt_len:]
        hyps_raw = decode_qwen_transcriptions(processor, generated)
        return refs_raw, paths, hyps_raw, None, generated

    if "input_features" not in batch or not torch.is_tensor(batch["input_features"]):
        raise KeyError("Batch must contain tensor 'input_features'")
    input_features = batch["input_features"].to(device)
    refs_raw = _as_str_list(batch.get("text"), int(input_features.size(0)))
    paths = _as_str_list(batch.get("audio_path"), int(input_features.size(0)))
    outputs = model.generate(input_features, **gen_kwargs)
    if with_scores:
        sequences = outputs.sequences
        hyps_raw = processor.batch_decode(sequences, skip_special_tokens=True)
        return refs_raw, paths, hyps_raw, outputs, sequences[:, 1:]
    hyps_raw = processor.batch_decode(outputs, skip_special_tokens=True)
    return refs_raw, paths, hyps_raw, None, outputs


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    processor,
    output_csv_path: Optional[str] = None,
    language: str = "en",
    task: str = "transcribe",
    desc: str = "evaluate",
    max_new_tokens: int = 256,
) -> Dict[str, float]:
    """Greedy-decode a dataloader and compute WER / Sub% / Ins% / Del%.

    Returns rates in percent relative to reference word count (N = H+S+D).
    """
    device = next(model.parameters()).device
    model.eval()
    qwen = is_qwen3_asr(processor)

    if not qwen:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        gen_kwargs = dict(
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            forced_decoder_ids=forced_decoder_ids,
        )
    else:
        gen_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
        )

    records: List[Dict[str, Any]] = []
    total_h = total_s = total_d = total_i = 0

    for batch in progress_bar(dataloader, total=len(dataloader), desc=desc, unit="batch"):
        try:
            refs_raw, paths, hyps_raw, _, _ = _decode_batch(
                model, processor, batch, device, gen_kwargs, qwen=qwen, with_scores=False
            )
        except (ValueError, RuntimeError, ZeroDivisionError) as exc:
            audio = batch.get("audio") if qwen else None
            n_audio = len(audio) if isinstance(audio, (list, tuple)) else 0
            if not qwen or n_audio <= 1:
                raise
            logger.warning(
                "Qwen batch of %d failed (%s: %s); decoding one utterance at a time",
                n_audio,
                type(exc).__name__,
                exc,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            refs_raw, paths, hyps_raw = [], [], []
            for index in range(n_audio):
                one_refs, one_paths, one_hyps, _, _ = _decode_batch(
                    model,
                    processor,
                    _slice_eval_batch(batch, index),
                    device,
                    gen_kwargs,
                    qwen=True,
                    with_scores=False,
                )
                refs_raw.extend(one_refs)
                paths.extend(one_paths)
                hyps_raw.extend(one_hyps)

        for path, ref_raw, hyp_raw in zip(paths, refs_raw, hyps_raw):
            ref = normalize_text(ref_raw)
            hyp = normalize_text(hyp_raw)
            h, s, d, i = _alignment_counts(ref, hyp)
            total_h += h
            total_s += s
            total_d += d
            total_i += i
            n_ref = h + s + d
            records.append(
                {
                    "audio_path": path,
                    "reference": ref_raw,
                    "reference_norm": ref,
                    "hypothesis": hyp_raw,
                    "hypothesis_norm": hyp,
                    "hits": h,
                    "substitutions": s,
                    "deletions": d,
                    "insertions": i,
                    "n_ref_words": n_ref,
                    "utt_wer": ((s + d + i) / n_ref) if n_ref > 0 else (0.0 if not hyp else 1.0),
                }
            )

    n_ref_total = total_h + total_s + total_d
    if n_ref_total == 0:
        wer = 0.0 if total_i == 0 else 100.0
        sub = del_ = 0.0
        ins = 100.0 if total_i > 0 else 0.0
    else:
        wer = 100.0 * (total_s + total_d + total_i) / n_ref_total
        sub = 100.0 * total_s / n_ref_total
        del_ = 100.0 * total_d / n_ref_total
        ins = 100.0 * total_i / n_ref_total

    metrics = {
        "wer": float(wer),
        "sub": float(sub),
        "ins": float(ins),
        "del": float(del_),
        "hits": float(total_h),
        "substitutions": float(total_s),
        "deletions": float(total_d),
        "insertions": float(total_i),
        "n_ref_words": float(n_ref_total),
        "n_utterances": float(len(records)),
    }
    logger.info(
        "Eval %s | WER=%.2f Sub=%.2f Ins=%.2f Del=%.2f (N=%d)",
        desc,
        metrics["wer"],
        metrics["sub"],
        metrics["ins"],
        metrics["del"],
        int(n_ref_total),
    )

    if output_csv_path:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(output_csv_path, index=False)
        logger.info("Wrote sentence-level predictions to %s", output_csv_path)

    return metrics


def _generation_token_mask(token_ids: torch.Tensor, eos_token_id: Optional[Any]) -> torch.Tensor:
    """Keep tokens up to and including the first EOS; drop subsequent pads."""
    keep = torch.ones_like(token_ids, dtype=torch.bool)
    if eos_token_id is None:
        return keep
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id]
    eos_ids = [int(x) for x in eos_ids if x is not None]
    if not eos_ids:
        return keep
    finished = torch.zeros(token_ids.size(0), dtype=torch.bool, device=token_ids.device)
    for step in range(token_ids.size(1)):
        keep[:, step] = ~finished
        step_eos = torch.zeros_like(finished)
        for eid in eos_ids:
            step_eos = step_eos | (token_ids[:, step] == eid)
        finished = finished | step_eos
    return keep


@torch.inference_mode()
def score_model_with_confidence(
    model: torch.nn.Module,
    dataloader,
    processor,
    language: str = "en",
    task: str = "transcribe",
    desc: str = "score-confidence",
    max_new_tokens: int = 256,
) -> pd.DataFrame:
    """Greedy-decode a dataloader and attach token-level confidence + correctness.

    Confidence is the mean generated-token log-prob after ``log_softmax``.
    An utterance is correct when Whisper-normalized WER is exactly 0.
    """
    device = next(model.parameters()).device
    model.eval()
    qwen = is_qwen3_asr(processor)

    if not qwen:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        gen_kwargs = dict(
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            forced_decoder_ids=forced_decoder_ids,
            return_dict_in_generate=True,
            output_scores=True,
        )
    else:
        gen_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            return_dict_in_generate=True,
            output_scores=True,
        )
    eos_token_id = getattr(model.config, "eos_token_id", None)
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0] if pad_token_id else None
    records: List[Dict[str, Any]] = []

    for batch in progress_bar(dataloader, total=len(dataloader), desc=desc, unit="batch"):
        refs_raw, paths, hyps_raw, outputs, gen_tokens = _decode_batch(
            model, processor, batch, device, gen_kwargs, qwen=qwen, with_scores=True
        )
        if getattr(outputs, "scores", None):
            if qwen:
                logprobs = []
                for step, score in enumerate(outputs.scores):
                    if step >= gen_tokens.shape[1]:
                        break
                    lp = torch.log_softmax(score.float(), dim=-1)
                    token_ids = gen_tokens[:, step].unsqueeze(1)
                    logprobs.append(lp.gather(1, token_ids).squeeze(1))
                transition_scores = (
                    torch.stack(logprobs, dim=1)
                    if logprobs
                    else torch.zeros_like(gen_tokens, dtype=torch.float32)
                )
            else:
                transition_scores = model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
            if transition_scores.shape[1] != gen_tokens.shape[1]:
                length = min(transition_scores.shape[1], gen_tokens.shape[1])
                transition_scores = transition_scores[:, :length]
                gen_tokens = gen_tokens[:, :length]
        else:
            transition_scores = torch.zeros_like(gen_tokens, dtype=torch.float32)

        token_mask = _generation_token_mask(gen_tokens, eos_token_id)
        if pad_token_id is not None:
            token_mask = token_mask & (gen_tokens != int(pad_token_id))
        token_counts = token_mask.sum(dim=1).clamp(min=1)
        masked_scores = transition_scores.masked_fill(~token_mask, 0.0)
        mean_lp = masked_scores.sum(dim=1) / token_counts.float()
        min_lp = transition_scores.masked_fill(~token_mask, float("inf")).min(dim=1).values
        min_lp = torch.where(torch.isfinite(min_lp), min_lp, torch.zeros_like(min_lp))

        for i, (path, ref_raw, hyp_raw) in enumerate(zip(paths, refs_raw, hyps_raw)):
            ref = normalize_text(ref_raw)
            hyp = normalize_text(hyp_raw)
            h, s, d, ins = _alignment_counts(ref, hyp)
            n_ref = h + s + d
            utt_wer = ((s + d + ins) / n_ref) if n_ref > 0 else (0.0 if not hyp else 1.0)
            n_tokens = int(token_mask[i].sum().item())
            records.append(
                {
                    "audio_path": path,
                    "reference": ref_raw,
                    "reference_norm": ref,
                    "hypothesis": hyp_raw,
                    "hypothesis_norm": hyp,
                    "hits": h,
                    "substitutions": s,
                    "deletions": d,
                    "insertions": ins,
                    "n_ref_words": n_ref,
                    "utt_wer": float(utt_wer),
                    "is_correct": bool(utt_wer == 0.0),
                    "mean_token_logprob": float(mean_lp[i].item()) if n_tokens else float("nan"),
                    "min_token_logprob": float(min_lp[i].item()) if n_tokens else float("nan"),
                    "n_gen_tokens": n_tokens,
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        "Scored %d utterances | correct=%d | mean_logprob=%.3f",
        len(df),
        int(df["is_correct"].sum()) if len(df) else 0,
        float(df["mean_token_logprob"].mean()) if len(df) else float("nan"),
    )
    return df


def build_eval_dataloader(
    csv_path: str,
    processor,
    batch_size: int = 8,
    text_column: str = "text",
    num_workers: int = 2,
    language: str = "en",
):
    """Build a DataLoader that yields padded features + raw text for evaluation."""
    from torch.utils.data import DataLoader

    from .dataset import L2SpeechDataset

    dataset = L2SpeechDataset(csv_path, processor=processor, text_column=text_column, language=language)
    qwen = is_qwen3_asr(processor)

    def collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if qwen:
            return {
                "audio": [f["audio"] for f in features],
                "text": [f["text"] for f in features],
                "audio_path": [f["audio_path"] for f in features],
                "language": language,
            }
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["text"] = [f["text"] for f in features]
        batch["audio_path"] = [f["audio_path"] for f in features]
        return batch

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)


def write_eval_summary_md(
    summary: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Target Evaluation Summary",
) -> None:
    """Write a markdown table suitable for paper drafts."""
    lines = [
        f"# {title}",
        "",
        "| Model | WER (%) | Sub (%) | Ins (%) | Del (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in summary.items():
        lines.append(
            f"| {name} | {m.get('wer', float('nan')):.2f} | "
            f"{m.get('sub', float('nan')):.2f} | "
            f"{m.get('ins', float('nan')):.2f} | "
            f"{m.get('del', float('nan')):.2f} |"
        )
    lines.append("")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote summary markdown to %s", output_path)


def write_eval_detailed_json(summary: Dict[str, Dict[str, float]], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote detailed metrics JSON to %s", output_path)
