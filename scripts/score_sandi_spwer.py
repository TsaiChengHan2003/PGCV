#!/usr/bin/env python3
"""Score a SANDI ASR CTM with the official SpWER pipeline.

Reads STM / GLM / sclite / GLM-filter tools from the read-only corpus.
Writes every intermediate and the sclite reports under ``--out_dir`` in this
repo. Does not modify ``/share/corpus``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CORPUS_DEFAULT = Path("/share/corpus/sandi2025-challenge")
SPLIT_DEFAULT = "eval"

SUM_AVG_RE = re.compile(
    r"\|\s*Sum/Avg\s*\|\s+(\d+)\s+(\d+)\s*\|\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def corpus_paths(corpus: Path) -> Dict[str, Path]:
    nested_wer = corpus / "scoring-tools" / "wer-tools" / "wer-tools"
    return {
        "corpus": corpus,
        "baseline_tools": corpus / "baseline-pipeline" / "tools",
        "wer_tools": nested_wer,
        "sclite": nested_wer / "sctk-2.4.9" / "bin" / "sclite",
        "hubscr": nested_wer / "hubscr07.sort.filt.pl",
        "glm": corpus / "reference-materials" / "glm" / "sandi-20241209.glm",
    }


def split_refs(corpus: Path, split: str) -> Dict[str, Path]:
    return {
        "flist": corpus / "reference-materials" / "flists.flac" / f"{split}-asr.tsv",
        "stm": corpus / "reference-materials" / "stms" / f"{split}-asr.stm",
        "official_ctm": corpus / "reference-materials" / "pre-norm" / f"{split}-asr.ctm",
    }


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
) -> None:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    out = (result.stdout or "") + (result.stderr or "")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(out, encoding="utf-8")
    if result.returncode != 0:
        sys.stderr.write(out)
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    if out.strip():
        print(out.rstrip(), flush=True)


def python_env(pythonpath_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pythonpath_dir)
    return env


def ensure_hubscr_cwd(out_dir: Path, wer_tools: Path) -> Path:
    """Layout hubscr07 expects: ./scoring-tools/wer-tools/{sctk,tranfilt,...}."""
    cwd = out_dir / "_hubscr_cwd"
    link_parent = cwd / "scoring-tools"
    link = link_parent / "wer-tools"
    link_parent.mkdir(parents=True, exist_ok=True)
    target = wer_tools.resolve()
    if link.is_symlink() or link.exists():
        if not (link.is_symlink() and link.resolve() == target):
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
            link.symlink_to(target)
    else:
        link.symlink_to(target)
    return cwd


def _fileid(token: str) -> str:
    return token.split("_PH")[0]


def stm_fileids(stm_path: Path) -> list[str]:
    ids: list[str] = []
    seen = set()
    for line in stm_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        items = line.split()
        if ";;" in items[0]:
            continue
        fileid = _fileid(items[0])
        if fileid not in seen:
            seen.add(fileid)
            ids.append(fileid)
    return ids


def read_ctm_rest_by_fileid(ctm_path: Path) -> Dict[str, list[str]]:
    info: Dict[str, list[str]] = {}
    for line in ctm_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        items = line.split()
        fileid = _fileid(items[0])
        info.setdefault(fileid, []).append(" ".join(items[1:]))
    return info


def sort_subset_ctm(input_ctm: Path, input_stm: Path, output_ctm: Path) -> list[str]:
    """Official ``sort_subset_ctm.py``, but missing STM fileids are empty hyps.

    The corpus tool does ``ctm_info[fileid]`` and KeyErrors when every token
    was dropped (decode produced no words, or ``ctm_norm`` stripped only
    hesitations such as ``uh``). An empty hyp is the correct SpWER outcome
    (all deletions). ``/share/corpus`` stays read-only.
    """
    ctm_info = read_ctm_rest_by_fileid(input_ctm)
    selected = sorted(stm_fileids(input_stm))
    missing = [fileid for fileid in selected if fileid not in ctm_info]
    output_ctm.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{fileid} {rest}" for fileid in selected for rest in ctm_info.get(fileid, [])]
    output_ctm.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    if missing:
        sample = ", ".join(missing[:8])
        more = "..." if len(missing) > 8 else ""
        print(
            f"WARNING: {len(missing)} STM fileids have no CTM tokens after norm "
            f"(scored as empty hyp): {sample}{more}",
            flush=True,
        )
    return missing


def parse_sum_avg(sys_text: str) -> Dict[str, Any]:
    match = SUM_AVG_RE.search(sys_text)
    if not match:
        raise ValueError("Could not parse sclite Sum/Avg line")
    n_snt, n_wrd, corr, sub, deletion, ins, err, s_err = match.groups()
    return {
        "n_sentences": int(n_snt),
        "n_words": int(n_wrd),
        "corr": float(corr),
        "sub": float(sub),
        "del": float(deletion),
        "ins": float(ins),
        "err": float(err),
        "s_err": float(s_err),
    }


def score_ctm(
    *,
    ctm: Path,
    out_dir: Path,
    corpus: Path,
    split: str,
    flist: Optional[Path] = None,
    stm: Optional[Path] = None,
    glm: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    paths = corpus_paths(corpus)
    refs = split_refs(corpus, split)
    flist = Path(flist) if flist else refs["flist"]
    stm = Path(stm) if stm else refs["stm"]
    glm = Path(glm) if glm else paths["glm"]

    for required in (ctm, flist, stm, glm, paths["sclite"], paths["hubscr"]):
        if not Path(required).exists():
            raise FileNotFoundError(required)

    if out_dir.exists():
        if not force:
            raise FileExistsError(f"Scoring directory exists (pass --force): {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)
    log_dir = out_dir / "LOGs"
    norm_dir = out_dir / "norm"
    ref_dir = out_dir / "ref"
    hyp_dir = out_dir / "hyp"
    for folder in (log_dir, norm_dir, ref_dir, hyp_dir):
        folder.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    pre_norm = norm_dir / "asr.pre-norm.ctm"
    norm_ctm = norm_dir / "asr.ctm"
    final_ctm = out_dir / "asr.ctm"

    run_cmd(
        [
            python,
            str(paths["baseline_tools"] / "select_files.py"),
            "--input_ctm",
            str(ctm),
            "--wav_list",
            str(flist),
            "--output_ctm",
            str(pre_norm),
        ],
        cwd=out_dir,
        env=python_env(paths["baseline_tools"]),
        log_path=log_dir / "select_files.log",
    )
    run_cmd(
        [
            python,
            str(paths["baseline_tools"] / "ctm_norm.py"),
            "--ctm_file",
            str(pre_norm),
            "--output_file",
            str(norm_ctm),
            "--task",
            "asr",
        ],
        cwd=out_dir,
        env=python_env(paths["baseline_tools"]),
        log_path=log_dir / "ctm_norm.log",
    )
    shutil.copy2(norm_ctm, final_ctm)

    hyp_ctm = hyp_dir / "asr.ctm"
    ref_stm = ref_dir / stm.name
    missing = sort_subset_ctm(final_ctm, stm, hyp_ctm)
    (log_dir / "sort_subset_ctm.log").write_text(
        json.dumps({"missing_empty_hyp": missing}, indent=2) + "\n",
        encoding="utf-8",
    )
    run_cmd(
        [
            python,
            str(paths["wer_tools"] / "mergescorestmsegs.py"),
            str(stm),
            str(ref_stm),
        ],
        cwd=out_dir,
        env=python_env(paths["wer_tools"]),
        log_path=log_dir / "mergescorestmsegs.log",
    )

    hubscr_cwd = ensure_hubscr_cwd(out_dir, paths["wer_tools"])
    run_cmd(
        [
            "perl",
            "./scoring-tools/wer-tools/hubscr07.sort.filt.pl",
            "-h",
            "rt-stt",
            "-l",
            "english",
            "-g",
            str(glm),
            "-r",
            str(ref_stm),
            "-k",
            str(hyp_ctm),
        ],
        cwd=hubscr_cwd,
        log_path=log_dir / "glm_filter.log",
    )

    ref_filt = Path(str(ref_stm) + ".filt")
    hyp_filt = Path(str(hyp_ctm) + ".filt")
    if not ref_filt.is_file() or not hyp_filt.is_file():
        raise FileNotFoundError(f"GLM filter outputs missing: {ref_filt}, {hyp_filt}")

    sclite_cmd = [
        str(paths["sclite"]),
        "-r",
        str(ref_filt),
        "stm",
        "-h",
        str(hyp_filt),
        "ctm",
        "-F",
        "-D",
        "-o",
        "sum",
        "rsum",
        "sgml",
        "lur",
        "dtl",
        "pra",
    ]
    run_cmd(sclite_cmd, cwd=out_dir, log_path=log_dir / "sclite.log")

    sys_path = Path(str(hyp_filt) + ".sys")
    if not sys_path.is_file():
        raise FileNotFoundError(f"sclite did not write {sys_path}")
    sys_text = sys_path.read_text(encoding="utf-8", errors="replace")
    summary = parse_sum_avg(sys_text)
    summary.update(
        {
            "split": split,
            "input_ctm": str(ctm),
            "normalized_ctm": str(final_ctm),
            "sys_path": str(sys_path),
            "corpus": str(corpus),
        }
    )
    summary_path = out_dir / "spwer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"SpWER {summary['err']:.2f}  "
        f"(Sub {summary['sub']:.2f} / Del {summary['del']:.2f} / Ins {summary['ins']:.2f})  "
        f"n={summary['n_sentences']} utt / {summary['n_words']} words",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Official SANDI SpWER scoring")
    parser.add_argument("--ctm", required=True, help="Hypothesis CTM (raw or already selected)")
    parser.add_argument("--out_dir", required=True, help="Writable scoring directory inside the repo")
    parser.add_argument("--split", default=SPLIT_DEFAULT, choices=("train", "dev", "eval"))
    parser.add_argument("--corpus", default=str(CORPUS_DEFAULT))
    parser.add_argument("--flist", default="", help="Override wav/id list (only IDs are used)")
    parser.add_argument("--stm", default="")
    parser.add_argument("--glm", default="")
    parser.add_argument("--force", action="store_true", help="Replace an existing --out_dir")
    args = parser.parse_args()

    score_ctm(
        ctm=Path(args.ctm).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        corpus=Path(args.corpus).resolve(),
        split=args.split,
        flist=Path(args.flist).resolve() if args.flist else None,
        stm=Path(args.stm).resolve() if args.stm else None,
        glm=Path(args.glm).resolve() if args.glm else None,
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
