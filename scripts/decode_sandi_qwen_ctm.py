#!/usr/bin/env python3
"""Full-file Qwen3-ASR decode → SANDI CTM (corpus read-only).

Qwen has no openai-whisper word timestamps. Words are spread uniformly
across the file duration so sclite can still overlap STM regions.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import soundfile as sf
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr_backend import (  # noqa: E402
    decode_qwen_transcriptions,
    load_asr_model,
    move_qwen_batch_to_device,
    qwen_language_name,
    qwen_model_inputs,
    require_qwen3_asr_transformers,
    to_qwen_audio_batch,
)
from src.audio_utils import load_and_preprocess_audio  # noqa: E402
from src.task_vector import apply_task_vector, load_tau  # noqa: E402

CORPUS_DEFAULT = Path("/share/corpus/sandi2025-challenge")


def write_abs_flist(src_flist: Path, corpus: Path, dest: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    missing = 0
    for line in src_flist.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        fileid, rel = line.split()
        wav = rel if os.path.isabs(rel) else str(corpus / rel)
        if not Path(wav).is_file():
            missing += 1
            print(f"WARNING: missing audio {wav}", flush=True)
            continue
        rows.append((fileid, wav))
    if not rows:
        raise FileNotFoundError(f"No existing audio paths from {src_flist}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(f"{i}\t{p}\n" for i, p in rows), encoding="utf-8")
    if missing:
        print(f"Dropped {missing} missing files; kept {len(rows)}", flush=True)
    return rows


def audio_duration_sec(path: str) -> float:
    info = sf.info(path)
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def text_to_ctm_lines(fileid: str, text: str, duration: float) -> List[str]:
    words = [w for w in str(text).replace(",", " ").split() if w]
    if not words:
        return []
    dur = max(float(duration), 0.02)
    step = dur / len(words)
    lines = []
    for i, word in enumerate(words):
        start = i * step
        lines.append(f"{fileid} 1 {start:.2f} {step:.2f} {word} 1.00")
    return lines


def load_qwen_processor(student_model: str, hf_ckpt: str):
    from transformers import AutoProcessor

    last_err = None
    for source in (hf_ckpt, student_model):
        if not source:
            continue
        try:
            return AutoProcessor.from_pretrained(source)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Could not load Qwen processor from {hf_ckpt or student_model}: {last_err}")


def load_qwen_model(
    *,
    student_model: str,
    hf_ckpt: str,
    tau_path: str,
    lambda_val: float,
    device: str,
):
    require_qwen3_asr_transformers()
    source = hf_ckpt or student_model
    print(f"Loading Qwen {source}", flush=True)
    if device != "cpu" and torch.cuda.is_available():
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")
    model = load_asr_model(source, device=torch_device)
    if tau_path and abs(float(lambda_val)) >= 1e-12:
        print(f"Applying tau {tau_path} λ={float(lambda_val):.3f}", flush=True)
        apply_task_vector(model, load_tau(tau_path), float(lambda_val))
    model.eval()
    processor = load_qwen_processor(student_model, hf_ckpt)
    return model, processor, next(model.parameters()).device


def transcribe_one(model, processor, wav_path: str, device: torch.device, language: str, max_new_tokens: int) -> str:
    waveform = load_and_preprocess_audio(wav_path, target_sr=16000)
    audio = to_qwen_audio_batch(waveform)
    inputs = processor.apply_transcription_request(audio=audio, language=qwen_language_name(language))
    inputs = move_qwen_batch_to_device(inputs, device, dtype=getattr(model, "dtype", None))
    model_inputs = qwen_model_inputs(inputs)
    prompt_len = model_inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
        )
    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    generated = sequences[:, prompt_len:]
    texts = decode_qwen_transcriptions(processor, generated)
    return str(texts[0]).strip() if texts else ""


def decode(
    *,
    corpus: Path,
    split: str,
    out_dir: Path,
    device: str,
    student_model: str = "Qwen/Qwen3-ASR-0.6B-hf",
    tau_path: str = "",
    lambda_val: float = 0.0,
    hf_ckpt: str = "",
    overwrite: bool = False,
    limit: int = 0,
    language: str = "en",
    max_new_tokens: int = 512,
    file_ids: Optional[Sequence[str]] = None,
) -> Path:
    src_flist = corpus / "reference-materials" / "flists.flac" / f"{split}-asr.tsv"
    if not src_flist.is_file():
        raise FileNotFoundError(src_flist)
    out_dir.mkdir(parents=True, exist_ok=True)
    flist_path = out_dir / f"{split}-asr.abs.tsv"
    rows = write_abs_flist(src_flist, corpus, flist_path)
    if file_ids is not None:
        wanted = {str(file_id) for file_id in file_ids}
        rows = [(fileid, path) for fileid, path in rows if fileid in wanted]
        if not rows:
            raise ValueError(f"No {split} files matched the requested file_ids")
    if limit > 0:
        rows = rows[:limit]
        print(f"Limiting decode to first {len(rows)} utterances", flush=True)

    ctm_path = out_dir / "asr.ctm"
    existing: List[str] = []
    if ctm_path.is_file() and not overwrite:
        existing = [ln for ln in ctm_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        done_ids = {ln.split()[0] for ln in existing if ln.split()}
        rows = [(i, p) for i, p in rows if i not in done_ids]
        print(f"Resume CTM: keep {len(existing)} lines, left {len(rows)} utts", flush=True)
    if not rows:
        print(f"Decode already complete -> {ctm_path}", flush=True)
        return ctm_path

    model, processor, dev = load_qwen_model(
        student_model=student_model,
        hf_ckpt=hf_ckpt,
        tau_path=tau_path,
        lambda_val=lambda_val,
        device=device,
    )
    new_lines: List[str] = []
    for fileid, wav_path in tqdm(rows, desc="qwen-ctm", unit="utt"):
        hyp = transcribe_one(model, processor, wav_path, dev, language, max_new_tokens)
        dur = audio_duration_sec(wav_path)
        new_lines.extend(text_to_ctm_lines(fileid, hyp, dur))

    all_lines = existing + new_lines
    all_lines.sort(key=lambda x: x.split()[0] if x.split() else x)
    ctm_path.write_text("\n".join(all_lines) + ("\n" if all_lines else ""), encoding="utf-8")
    print(f"Decode done: new={len(rows)} ctm_lines={len(all_lines)} -> {ctm_path}", flush=True)
    return ctm_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SANDI full-file Qwen decode to CTM")
    parser.add_argument("--corpus", default=str(CORPUS_DEFAULT))
    parser.add_argument("--split", default="eval", choices=("train", "dev", "eval"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--student_model", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--tau", default="")
    parser.add_argument("--lambda_val", type=float, default=0.0)
    parser.add_argument("--hf_ckpt", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    decode(
        corpus=Path(args.corpus).resolve(),
        split=args.split,
        out_dir=Path(args.out_dir).resolve(),
        device=args.device,
        student_model=args.student_model,
        tau_path=args.tau,
        lambda_val=float(args.lambda_val),
        hf_ckpt=args.hf_ckpt,
        overwrite=bool(args.overwrite),
        limit=int(args.limit),
        max_new_tokens=int(args.max_new_tokens),
    )


if __name__ == "__main__":
    main()
