"""Offline diagnostics for verbatim L2 ASR: error types, loops, energy PSR.

Training / decoding never call this module. Pause-to-speech ratio is an
analysis-only energy VAD; it is not a disfluency tag.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)

PROFICIENCY_ORDER = {
    "pre-A": 0,
    "A1": 1,
    "A1+": 1.5,
    "A1A2": 2,
    "A2": 3,
    "A2+": 3.5,
    "A2B1": 4,
    "B1": 5,
    "B1+": 5.5,
    "B1B2": 6,
    "B2": 7,
    "B2+": 7.5,
    "C1": 8,
    "C1+": 8.5,
    "C2": 9,
}

GEMINI_PSR_BINS = (
    ("psr_lt_20", 0.0, 0.20),
    ("psr_20_40", 0.20, 0.40),
    ("psr_ge_40", 0.40, 1.01),
)


def proficiency_sort_key(level: Any) -> Tuple[float, str]:
    text = str(level).strip()
    return (float(PROFICIENCY_ORDER.get(text, 999)), text)


def tokenize_norm(text: Any) -> List[str]:
    return [tok for tok in str(text or "").strip().split() if tok]


def max_tiled_repeats(tokens: Sequence[str], min_k: int = 2, max_k: int = 8) -> Tuple[int, str]:
    """Count consecutive non-overlapping copies of the same k-gram (k>=2)."""
    n = len(tokens)
    if n < min_k * 2:
        return 0, ""
    best_n, best_p = 1, ""
    upper = min(int(max_k), n // 2)
    for k in range(int(min_k), upper + 1):
        i = 0
        while i + 2 * k <= n:
            block = tokens[i : i + k]
            run = 1
            j = i + k
            while j + k <= n and tokens[j : j + k] == block:
                run += 1
                j += k
            if run > best_n:
                best_n, best_p = run, " ".join(block)
            i += 1 if run == 1 else (run * k)
    return int(best_n), best_p


def max_ngram_run(tokens: Sequence[str], n: int = 1) -> Tuple[int, str]:
    if n < 1 or len(tokens) < n:
        return 0, ""
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    best_n, best_g, cur_n, prev = 1, grams[0], 1, grams[0]
    for gram in grams[1:]:
        if gram == prev:
            cur_n += 1
            if cur_n > best_n:
                best_n, best_g = cur_n, gram
        else:
            cur_n, prev = 1, gram
    return int(best_n), best_g


def rates_from_counts(
    *,
    total_h: float,
    total_s: float,
    total_d: float,
    total_i: float,
    n_utterances: int = 0,
) -> Dict[str, float]:
    n_ref = float(total_h + total_s + total_d)
    if n_ref <= 0:
        wer = 0.0 if total_i <= 0 else 100.0
        sub = del_ = 0.0
        ins = 100.0 if total_i > 0 else 0.0
    else:
        wer = 100.0 * (total_s + total_d + total_i) / n_ref
        sub = 100.0 * total_s / n_ref
        del_ = 100.0 * total_d / n_ref
        ins = 100.0 * total_i / n_ref
    return {
        "wer": float(wer),
        "sub": float(sub),
        "ins": float(ins),
        "del": float(del_),
        "hits": float(total_h),
        "substitutions": float(total_s),
        "deletions": float(total_d),
        "insertions": float(total_i),
        "n_ref_words": float(n_ref),
        "n_utterances": float(n_utterances),
    }


def metrics_from_pred_df(pred_df: pd.DataFrame) -> Dict[str, float]:
    if pred_df is None or len(pred_df) == 0:
        return rates_from_counts(total_h=0, total_s=0, total_d=0, total_i=0, n_utterances=0)
    return rates_from_counts(
        total_h=float(pred_df["hits"].sum()),
        total_s=float(pred_df["substitutions"].sum()),
        total_d=float(pred_df["deletions"].sum()),
        total_i=float(pred_df["insertions"].sum()),
        n_utterances=int(len(pred_df)),
    )


def annotate_loops(
    df: pd.DataFrame,
    *,
    loop_run_min: int = 8,
    phrase_run_min: int = 6,
    eos_hyp_words: int = 80,
    hyp_ref_ratio: float = 2.5,
) -> pd.DataFrame:
    """Add loop / failed-stop proxies from normalized text.

    Prediction CSVs have no decoder token ids, so ``likely_no_eos`` is a
    text-side stand-in: a long unigram run or a very long hypothesis.
    """
    out = df.copy()
    hyp_tokens = out["hypothesis_norm"].map(tokenize_norm)
    ref_tokens = out["reference_norm"].map(tokenize_norm)
    hyp_run = hyp_tokens.map(lambda toks: max_ngram_run(toks, 1))
    ref_run = ref_tokens.map(lambda toks: max_ngram_run(toks, 1))
    hyp_bi = hyp_tokens.map(lambda toks: max_ngram_run(toks, 2))
    hyp_phrase = hyp_tokens.map(lambda toks: max_tiled_repeats(toks, min_k=2, max_k=8))
    out["n_hyp_words"] = hyp_tokens.map(len)
    out["n_ref_words_tok"] = ref_tokens.map(len)
    out["max_unigram_run_hyp"] = hyp_run.map(lambda x: x[0])
    out["max_unigram_token_hyp"] = hyp_run.map(lambda x: x[1])
    out["max_unigram_run_ref"] = ref_run.map(lambda x: x[0])
    out["max_unigram_token_ref"] = ref_run.map(lambda x: x[1])
    out["max_bigram_run_hyp"] = hyp_bi.map(lambda x: x[0])
    out["max_bigram_token_hyp"] = hyp_bi.map(lambda x: x[1])
    out["max_phrase_run_hyp"] = hyp_phrase.map(lambda x: x[0])
    out["max_phrase_token_hyp"] = hyp_phrase.map(lambda x: x[1])
    out["hyp_ref_ratio"] = np.where(
        out["n_ref_words_tok"] > 0,
        out["n_hyp_words"] / out["n_ref_words_tok"].clip(lower=1),
        np.where(out["n_hyp_words"] > 0, np.inf, 0.0),
    )
    out["is_loop_utt"] = out["max_unigram_run_hyp"] >= int(loop_run_min)
    out["is_phrase_loop"] = out["max_phrase_run_hyp"] >= int(phrase_run_min)
    out["is_any_loop"] = out["is_loop_utt"] | out["is_phrase_loop"]
    out["is_extra_loop"] = out["is_any_loop"] & (
        (out["max_unigram_run_hyp"] > (out["max_unigram_run_ref"] + 1))
        | (out["max_phrase_run_hyp"] >= int(phrase_run_min))
    )
    out["likely_no_eos"] = out["is_any_loop"] | (out["n_hyp_words"] >= int(eos_hyp_words)) | (
        out["hyp_ref_ratio"] >= float(hyp_ref_ratio)
    )
    return out


def loop_summary(df: pd.DataFrame) -> Dict[str, float]:
    n = int(len(df))
    if n == 0:
        return {
            "n_utterances": 0.0,
            "n_loop_utt": 0.0,
            "pct_loop_utt": 0.0,
            "n_phrase_loop": 0.0,
            "pct_phrase_loop": 0.0,
            "n_any_loop": 0.0,
            "pct_any_loop": 0.0,
            "n_extra_loop": 0.0,
            "pct_extra_loop": 0.0,
            "n_likely_no_eos": 0.0,
            "pct_likely_no_eos": 0.0,
            "insertions_all": 0.0,
            "insertions_from_loop": 0.0,
            "pct_insertions_from_loop": 0.0,
            "insertions_from_any_loop": 0.0,
            "pct_insertions_from_any_loop": 0.0,
            "mean_max_unigram_run_hyp": 0.0,
            "mean_max_unigram_run_ref": 0.0,
            "mean_max_bigram_run_hyp": 0.0,
            "mean_max_phrase_run_hyp": 0.0,
            "median_hyp_ref_ratio": 0.0,
        }
    loop = df[df["is_loop_utt"]]
    phrase = df[df["is_phrase_loop"]]
    any_loop = df[df["is_any_loop"]]
    extra = df[df["is_extra_loop"]]
    no_eos = df[df["likely_no_eos"]]
    ins_all = float(df["insertions"].sum())
    ins_loop = float(loop["insertions"].sum()) if len(loop) else 0.0
    ins_any = float(any_loop["insertions"].sum()) if len(any_loop) else 0.0
    return {
        "n_utterances": float(n),
        "n_loop_utt": float(len(loop)),
        "pct_loop_utt": 100.0 * len(loop) / n,
        "n_phrase_loop": float(len(phrase)),
        "pct_phrase_loop": 100.0 * len(phrase) / n,
        "n_any_loop": float(len(any_loop)),
        "pct_any_loop": 100.0 * len(any_loop) / n,
        "n_extra_loop": float(len(extra)),
        "pct_extra_loop": 100.0 * len(extra) / n,
        "n_likely_no_eos": float(len(no_eos)),
        "pct_likely_no_eos": 100.0 * len(no_eos) / n,
        "insertions_all": ins_all,
        "insertions_from_loop": ins_loop,
        "pct_insertions_from_loop": (100.0 * ins_loop / ins_all) if ins_all > 0 else 0.0,
        "insertions_from_any_loop": ins_any,
        "pct_insertions_from_any_loop": (100.0 * ins_any / ins_all) if ins_all > 0 else 0.0,
        "mean_max_unigram_run_hyp": float(df["max_unigram_run_hyp"].mean()),
        "mean_max_unigram_run_ref": float(df["max_unigram_run_ref"].mean()),
        "mean_max_bigram_run_hyp": float(df["max_bigram_run_hyp"].mean()),
        "mean_max_phrase_run_hyp": float(df["max_phrase_run_hyp"].mean()),
        "median_hyp_ref_ratio": float(df["hyp_ref_ratio"].replace([np.inf, -np.inf], np.nan).median()),
    }


def top_loop_tokens(df: pd.DataFrame, k: int = 8) -> List[Dict[str, Any]]:
    looped = df[df["is_loop_utt"]].copy()
    if looped.empty:
        return []
    counts = (
        looped.groupby("max_unigram_token_hyp", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(k)
    )
    return [{"token": str(tok), "n_utt": int(n)} for tok, n in counts.items()]


def load_mono_waveform(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load mono float64 audio without torch. Same channel-mean + resample idea as training."""
    waveform_np, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
    waveform = waveform_np.mean(axis=1) if waveform_np.shape[1] > 1 else waveform_np[:, 0]
    if int(sample_rate) != int(target_sr):
        import librosa

        waveform = librosa.resample(waveform, orig_sr=int(sample_rate), target_sr=int(target_sr))
        sample_rate = int(target_sr)
    return waveform.astype(np.float64), int(sample_rate)


def _frame_rms(waveform: np.ndarray, sr: int, win_ms: float, hop_ms: float) -> Tuple[np.ndarray, int]:
    win = max(int(sr * win_ms / 1000.0), 1)
    hop = max(int(sr * hop_ms / 1000.0), 1)
    if waveform.size < win:
        energy = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
        return np.array([energy], dtype=np.float64), hop
    window = np.ones(win, dtype=np.float64) / float(win)
    conv = np.convolve(np.square(waveform), window, mode="valid")
    rms = np.sqrt(conv[::hop])
    return rms.astype(np.float64), hop


def _runs_from_mask(mask: np.ndarray) -> List[Tuple[int, int, bool]]:
    """Return (start, end_exclusive, value) runs over a 1-D bool mask."""
    if mask.size == 0:
        return []
    runs: List[Tuple[int, int, bool]] = []
    start = 0
    cur = bool(mask[0])
    for i in range(1, mask.size):
        val = bool(mask[i])
        if val != cur:
            runs.append((start, i, cur))
            start = i
            cur = val
    runs.append((start, mask.size, cur))
    return runs


def pause_to_speech_ratio(
    audio_path: str,
    *,
    sr: int = 16000,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    speech_db: float = 30.0,
    abs_rms_floor: float = 1e-4,
    pause_min_sec: float = 0.3,
) -> Dict[str, float]:
    """Energy VAD PSR. Leading/trailing silence is excluded from *internal* PSR."""
    waveform, meas_sr = load_mono_waveform(audio_path, target_sr=sr)
    duration = float(waveform.size / meas_sr) if meas_sr else 0.0
    sr = meas_sr
    rms, hop = _frame_rms(waveform, sr, win_ms, hop_ms)
    peak = float(np.max(rms)) if rms.size else 0.0
    rel_floor = peak * (10.0 ** (-float(speech_db) / 20.0))
    speech = rms > max(rel_floor, float(abs_rms_floor))
    frame_sec = float(hop) / float(sr)
    runs = _runs_from_mask(speech)

    def _dur(start: int, end: int) -> float:
        return max(end - start, 0) * frame_sec

    pause_all = sum(_dur(s, e) for s, e, is_speech in runs if not is_speech)
    internal = list(runs)
    if internal and not internal[0][2]:
        internal = internal[1:]
    if internal and not internal[-1][2]:
        internal = internal[:-1]
    pause_internal = sum(
        _dur(s, e) for s, e, is_speech in internal if (not is_speech) and _dur(s, e) >= float(pause_min_sec)
    )
    n_internal = sum(
        1 for s, e, is_speech in internal if (not is_speech) and _dur(s, e) >= float(pause_min_sec)
    )
    speech_dur = sum(_dur(s, e) for s, e, is_speech in runs if is_speech)
    return {
        "duration_sec_meas": duration,
        "speech_sec": float(speech_dur),
        "pause_all_sec": float(pause_all),
        "pause_internal_sec": float(pause_internal),
        "n_internal_pauses": float(n_internal),
        "psr_all": float(pause_all / duration) if duration > 0 else float("nan"),
        "psr_internal": float(pause_internal / duration) if duration > 0 else float("nan"),
        "peak_rms": peak,
    }


def gemini_psr_bin(psr: Any) -> str:
    try:
        value = float(psr)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(value):
        return "nan"
    if value < 0.20:
        return "psr_lt_20"
    if value < 0.40:
        return "psr_20_40"
    return "psr_ge_40"


def within_group_tertiles(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.nunique() < 2 or len(valid) < 3:
        out = pd.Series(["unbinned"] * len(series), index=series.index, dtype="object")
        out.loc[numeric.isna()] = "nan"
        return out
    try:
        cat = pd.qcut(valid, 3, duplicates="drop")
    except ValueError:
        labelled = pd.Series(["unbinned"] * len(valid), index=valid.index, dtype="object")
    else:
        n_bins = int(cat.cat.categories.size)
        names = {0: "T1_low", 1: "T2_mid", 2: "T3_high"}
        if n_bins == 3:
            labelled = cat.cat.codes.map(names).astype(str)
        elif n_bins == 2:
            labelled = cat.cat.codes.map({0: "T1_low", 1: "T3_high"}).astype(str)
        else:
            labelled = pd.Series(["unbinned"] * len(valid), index=valid.index, dtype="object")
    out = pd.Series(["nan"] * len(series), index=series.index, dtype="object")
    out.loc[labelled.index] = labelled.astype(str)
    out.loc[numeric.isna()] = "nan"
    return out


def attach_psr_bins(df: pd.DataFrame, psr_col: str = "psr_internal") -> pd.DataFrame:
    out = df.copy()
    out["psr_gemini_bin"] = out[psr_col].map(gemini_psr_bin)
    parts: List[pd.Series] = []
    for _, sub in out.groupby("proficiency_level", dropna=False):
        parts.append(within_group_tertiles(sub[psr_col]))
    out["psr_level_tertile"] = pd.concat(parts).reindex(out.index)
    return out


def iter_audio_paths(paths: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for raw in paths:
        path = str(raw)
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered
