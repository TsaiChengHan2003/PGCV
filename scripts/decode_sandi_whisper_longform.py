#!/usr/bin/env python3
"""Official-style Whisper long-form decode for SANDI ASR.

Matches ``baseline-pipeline/asr-tools/whisper_transcribe.py`` flags:
word timestamps, beam 5, suppress_blank, no condition-on-previous, English.

Writes JSON + CTM under ``--out_dir`` in this repo. Audio is read from the
corpus; nothing under ``/share/corpus`` is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CORPUS_DEFAULT = Path("/share/corpus/sandi2025-challenge")
ASR_TOOLS = CORPUS_DEFAULT / "baseline-pipeline" / "asr-tools"


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


def json_to_ctm(json_dir: Path, output_ctm: Path, asr_tools: Path) -> int:
    sys.path.insert(0, str(asr_tools))
    from whisper_json import json_to_ctm_lines  # type: ignore  # noqa: E402

    lines = json_to_ctm_lines(str(json_dir))
    lines.sort(key=lambda x: x.split()[0])
    output_ctm.parent.mkdir(parents=True, exist_ok=True)
    output_ctm.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def load_decode_model(
    *,
    model_name: str,
    device: str,
    model_dir: str,
    student_model: str,
    tau_path: str,
    lambda_val: float,
    hf_ckpt: str = "",
):
    sys.path.insert(0, str(PROJECT_ROOT))
    if hf_ckpt:
        from src.whisper_openai_bridge import load_openai_whisper_from_hf  # noqa: E402

        print(f"Loading HF checkpoint {hf_ckpt} into openai-whisper {model_name}", flush=True)
        return load_openai_whisper_from_hf(
            hf_model=hf_ckpt,
            openai_name=model_name,
            device=device,
            model_dir=model_dir,
        )
    if tau_path and abs(float(lambda_val)) >= 1e-12:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.whisper_openai_bridge import load_openai_whisper_with_tau  # noqa: E402

        print(
            f"Loading HF {student_model} + tau {tau_path} λ={float(lambda_val):.3f} "
            f"into openai-whisper {model_name}",
            flush=True,
        )
        return load_openai_whisper_with_tau(
            openai_name=model_name,
            student_model=student_model,
            tau_path=tau_path,
            lambda_val=float(lambda_val),
            device=device,
            model_dir=model_dir,
        )
    import whisper

    print(f"Loading openai-whisper model={model_name} device={device}", flush=True)
    return whisper.load_model(model_name, device=device, download_root=model_dir or None)


def decode(
    *,
    corpus: Path,
    split: str,
    out_dir: Path,
    model_name: str,
    device: str,
    beam_size: int,
    overwrite: bool,
    limit: int,
    model_dir: str,
    student_model: str = "openai/whisper-base",
    tau_path: str = "",
    lambda_val: float = 0.0,
    hf_ckpt: str = "",
    model=None,
    file_ids: Optional[Sequence[str]] = None,
) -> Path:
    src_flist = corpus / "reference-materials" / "flists.flac" / f"{split}-asr.tsv"
    asr_tools = corpus / "baseline-pipeline" / "asr-tools"
    if not src_flist.is_file():
        raise FileNotFoundError(src_flist)
    if not asr_tools.is_dir():
        raise FileNotFoundError(asr_tools)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
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

    if model is None:
        model = load_decode_model(
            model_name=model_name,
            device=device,
            model_dir=model_dir,
            student_model=student_model,
            tau_path=tau_path,
            lambda_val=lambda_val,
            hf_ckpt=hf_ckpt,
        )
    fp16 = device != "cpu" and next(model.parameters()).device.type == "cuda"

    n_skip = 0
    n_run = 0
    progress_path = out_dir / "progress.txt"
    for fileid, wav_path in tqdm(rows, desc="whisper-longform", unit="utt"):
        tgt = json_dir / f"{fileid}.json"
        if tgt.is_file() and not overwrite:
            n_skip += 1
            continue
        with torch.no_grad():
            result = model.transcribe(
                wav_path,
                without_timestamps=False,
                beam_size=beam_size,
                no_speech_threshold=0.6,
                word_timestamps=True,
                condition_on_previous_text=False,
                length_penalty=None,
                initial_prompt=None,
                fp16=fp16,
                language="en",
                suppress_blank=True,
            )
        for seg in result.get("segments", []):
            seg["attn_weights"] = 0
        with tgt.open("w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False)
        n_run += 1
        if (n_run + n_skip) % 10 == 0 or n_run == 1:
            progress_path.write_text(
                f"done={n_run + n_skip} ran={n_run} skipped={n_skip} total={len(rows)}\n",
                encoding="utf-8",
            )

    ctm_path = out_dir / "asr.ctm"
    n_lines = json_to_ctm(json_dir, ctm_path, asr_tools)
    print(
        f"Decode done: ran={n_run} skipped={n_skip} json={len(list(json_dir.glob('*.json')))} "
        f"ctm_lines={n_lines} -> {ctm_path}",
        flush=True,
    )
    return ctm_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SANDI long-form Whisper decode")
    parser.add_argument("--corpus", default=str(CORPUS_DEFAULT))
    parser.add_argument("--split", default="eval", choices=("train", "dev", "eval"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="base", help="openai-whisper model name (base, small, ...)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Decode only the first N utts (0 = all)")
    parser.add_argument("--model_dir", default="", help="whisper download_root (optional)")
    parser.add_argument("--student_model", default="openai/whisper-base", help="HF id used when applying τ")
    parser.add_argument("--tau", default="", help="Path to tau_*.pt (HF state-dict keys)")
    parser.add_argument("--lambda_val", type=float, default=0.0)
    parser.add_argument(
        "--hf_ckpt",
        default="",
        help="HF Whisper directory (model.safetensors). Copied into openai-whisper long-form decode.",
    )
    args = parser.parse_args()

    decode(
        corpus=Path(args.corpus).resolve(),
        split=args.split,
        out_dir=Path(args.out_dir).resolve(),
        model_name=args.model,
        device=args.device,
        beam_size=int(args.beam_size),
        overwrite=bool(args.overwrite),
        limit=int(args.limit),
        model_dir=args.model_dir,
        student_model=args.student_model,
        tau_path=args.tau,
        lambda_val=float(args.lambda_val),
        hf_ckpt=args.hf_ckpt,
    )


if __name__ == "__main__":
    main()
