"""Speaker-level audio embeddings and k-means gate for per-prototype tau.

Train-only fit: centroids come from A01 train speakers. Dev/test/A02 speakers
are assigned by nearest centroid (speaker-disjoint splits never appear in fit).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .asr_backend import (
    is_qwen3_asr,
    move_qwen_batch_to_device,
    qwen_model_inputs,
    to_qwen_audio_batch,
)
from .progress import progress_bar

logger = logging.getLogger(__name__)

SPEAKER_COL = "global_speaker_id"


def l2_normalize_np(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def _whisper_pooled_encoder(model: torch.nn.Module, input_features: torch.Tensor) -> torch.Tensor:
    encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.model.encoder
    hidden = encoder(input_features).last_hidden_state
    return F.normalize(hidden.mean(dim=1).float(), dim=-1)


def _qwen_audio_forward_kwargs(model_inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Keep only audio-tower inputs.

    The full transcription batch also has LM ``attention_mask``. Passing that
    through ``get_audio_features(**batch)`` collides with the encoder's own
    ``attention_mask=None`` and raises TypeError in sdpa_attention_forward.
    """
    if "input_features" not in model_inputs:
        raise KeyError("Qwen batch missing input_features for speaker embeddings")
    payload = {"input_features": model_inputs["input_features"]}
    mask = model_inputs.get("input_features_mask")
    if mask is not None:
        payload["input_features_mask"] = mask
    return payload


def _post_cnn_length(lengths: torch.Tensor) -> torch.Tensor:
    out = lengths.to(torch.long)
    for _ in range(3):
        out = torch.where(out > 0, (out - 1) // 2 + 1, torch.zeros_like(out))
    return out


def _qwen_packed_token_counts(mask: torch.Tensor, n_window: int) -> torch.Tensor:
    batch, padded = mask.shape
    chunk_len = int(n_window) * 2
    if chunk_len <= 0 or padded % chunk_len != 0:
        return _post_cnn_length(mask.sum(dim=-1))
    num_chunks = padded // chunk_len
    chunk_lens = mask.view(batch, num_chunks, chunk_len).sum(dim=-1)
    return _post_cnn_length(chunk_lens).sum(dim=-1)


def _pool_qwen_audio_hidden(
    hidden: torch.Tensor,
    mask: Optional[torch.Tensor],
    n_window: Optional[int],
) -> torch.Tensor:
    if hidden.ndim == 3:
        return F.normalize(hidden.mean(dim=1).float(), dim=-1)
    if hidden.ndim != 2:
        raise RuntimeError(f"Unexpected Qwen audio hidden shape {tuple(hidden.shape)}")
    if mask is None or n_window is None:
        return F.normalize(hidden.mean(dim=0, keepdim=True).float(), dim=-1)
    counts = [max(int(x), 0) for x in _qwen_packed_token_counts(mask, n_window).tolist()]
    if sum(counts) != hidden.size(0):
        if mask.size(0) == 1:
            return F.normalize(hidden.mean(dim=0, keepdim=True).float(), dim=-1)
        raise RuntimeError(
            f"Packed Qwen audio tokens {hidden.size(0)} do not match mask lengths {counts}"
        )
    pooled = []
    offset = 0
    width = hidden.size(-1)
    for count in counts:
        chunk = hidden[offset : offset + count] if count else hidden.new_zeros((1, width))
        offset += count
        pooled.append(chunk.mean(dim=0))
    return F.normalize(torch.stack(pooled, dim=0).float(), dim=-1)


def _qwen_audio_hidden(model: torch.nn.Module, model_inputs: Dict[str, Any]) -> torch.Tensor:
    """Return L2-normalized utterance embeddings ``[B, D]`` from the audio tower."""
    audio_inputs = _qwen_audio_forward_kwargs(model_inputs)
    hidden = None
    if hasattr(model, "get_audio_features"):
        feats = model.get_audio_features(**audio_inputs)
        if torch.is_tensor(feats):
            hidden = feats
        else:
            hidden = getattr(feats, "last_hidden_state", None)
    if hidden is None:
        inner = getattr(model, "model", model)
        for attr in ("audio_tower", "audio_encoder", "encoder"):
            tower = getattr(inner, attr, None)
            if tower is None:
                continue
            out = tower(**audio_inputs) if "input_features_mask" in audio_inputs else tower(audio_inputs["input_features"])
            hidden = getattr(out, "last_hidden_state", out)
            if torch.is_tensor(hidden):
                break
    if hidden is None or not torch.is_tensor(hidden):
        raise RuntimeError("Could not locate a Qwen3-ASR audio encoder for speaker embeddings")
    tower = getattr(getattr(model, "model", model), "audio_tower", None)
    n_window = getattr(tower, "n_window", None)
    return _pool_qwen_audio_hidden(hidden, audio_inputs.get("input_features_mask"), n_window)


@torch.inference_mode()
def extract_utterance_embeddings(
    model: torch.nn.Module,
    dataloader,
    processor,
    desc: str = "speaker-embed",
) -> pd.DataFrame:
    """Return one L2-normalized encoder embedding per utterance."""
    device = next(model.parameters()).device
    model.eval()
    qwen = is_qwen3_asr(processor)
    rows: List[Dict[str, Any]] = []

    for batch in progress_bar(dataloader, total=len(dataloader), desc=desc, unit="batch"):
        paths: Sequence[str] = batch.get("audio_path", [])
        if isinstance(paths, str):
            paths = [paths]
        if qwen:
            audio = to_qwen_audio_batch(batch["audio"])
            language = batch.get("language", "en")
            inputs = processor.apply_transcription_request(audio=audio, language=language)
            inputs = move_qwen_batch_to_device(inputs, device, dtype=getattr(model, "dtype", None))
            model_inputs = qwen_model_inputs(inputs)
            pooled = _qwen_audio_hidden(model, model_inputs)
        else:
            if "input_features" not in batch:
                raise KeyError("Whisper embedding batch must contain input_features")
            pooled = _whisper_pooled_encoder(model, batch["input_features"].to(device))
        cpu = pooled.detach().cpu().numpy()
        for path, vec in zip(paths, cpu):
            rows.append({"audio_path": path, "embedding": vec.astype(np.float32)})
    return pd.DataFrame(rows)


def pool_speaker_embeddings(
    utt_emb: pd.DataFrame,
    meta: pd.DataFrame,
    speaker_col: str = SPEAKER_COL,
) -> pd.DataFrame:
    """Mean-pool utterance embeddings to one L2-normalized vector per speaker."""
    cols = ["audio_path", speaker_col]
    if "proficiency_level" in meta.columns:
        cols.append("proficiency_level")
    merged = utt_emb.merge(meta[cols], on="audio_path", how="inner")
    if merged.empty:
        raise ValueError("No overlapping audio_path between embeddings and metadata")
    records: List[Dict[str, Any]] = []
    for speaker, sub in merged.groupby(speaker_col, sort=False):
        stacked = np.stack(sub["embedding"].to_list(), axis=0)
        mean_vec = l2_normalize_np(stacked.mean(axis=0, keepdims=True))[0]
        rec: Dict[str, Any] = {
            speaker_col: str(speaker),
            "n_utterances": int(len(sub)),
            "embedding": mean_vec.astype(np.float32),
        }
        if "proficiency_level" in sub.columns:
            mode = sub["proficiency_level"].astype(str).mode()
            rec["proficiency_level"] = str(mode.iloc[0]) if len(mode) else ""
        records.append(rec)
    out = pd.DataFrame(records)
    logger.info("Pooled %d utterance embeddings -> %d speakers", len(merged), len(out))
    return out


def embedding_matrix(speaker_df: pd.DataFrame) -> np.ndarray:
    return np.stack(speaker_df["embedding"].to_list(), axis=0).astype(np.float32)


def kmeans_fit(
    features: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    n_iter: int = 100,
    n_init: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means++ with several random inits. Returns (centroids[k, d], labels[n])."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    x = np.asarray(features, dtype=np.float64)
    n, dim = x.shape
    k = min(int(n_clusters), n)
    rng_master = np.random.RandomState(seed)

    best_inertia = np.inf
    best_centroids = None
    best_labels = None
    for init_i in range(n_init):
        rng = np.random.RandomState(int(rng_master.randint(0, 2**31 - 1)))
        centroids = _kmeans_plus_plus(x, k, rng)
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(n_iter):
            dists = _pairwise_sq_dists(x, centroids)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            centroids = _recompute_centroids(x, labels, k, rng)
        inertia = float(_pairwise_sq_dists(x, centroids)[np.arange(n), labels].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.copy()
            best_labels = labels.copy()
    assert best_centroids is not None and best_labels is not None
    logger.info("k-means k=%d n=%d inertia=%.4f", k, n, best_inertia)
    return best_centroids.astype(np.float32), best_labels


def _kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    n, dim = x.shape
    centroids = np.empty((k, dim), dtype=np.float64)
    first = int(rng.randint(0, n))
    centroids[0] = x[first]
    closest = _pairwise_sq_dists(x, centroids[:1]).reshape(-1)
    for i in range(1, k):
        weights = np.clip(closest, 1e-12, None)
        weights = weights / weights.sum()
        idx = int(rng.choice(n, p=weights))
        centroids[i] = x[idx]
        d_new = _pairwise_sq_dists(x, centroids[i : i + 1]).reshape(-1)
        closest = np.minimum(closest, d_new)
    return centroids


def _pairwise_sq_dists(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    x_norm = (x * x).sum(axis=1, keepdims=True)
    c_norm = (centroids * centroids).sum(axis=1, keepdims=True).T
    return np.clip(x_norm + c_norm - 2.0 * x.dot(centroids.T), 0.0, None)


def _recompute_centroids(
    x: np.ndarray,
    labels: np.ndarray,
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    dim = x.shape[1]
    centroids = np.zeros((k, dim), dtype=np.float64)
    for cluster in range(k):
        mask = labels == cluster
        if not np.any(mask):
            centroids[cluster] = x[int(rng.randint(0, len(x)))]
        else:
            centroids[cluster] = x[mask].mean(axis=0)
    return centroids


def assign_nearest(
    features: np.ndarray,
    centroids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, euclidean distances to assigned centroid)."""
    dists = np.sqrt(_pairwise_sq_dists(np.asarray(features, dtype=np.float64), np.asarray(centroids, dtype=np.float64)))
    labels = dists.argmin(axis=1).astype(np.int64)
    assigned = dists[np.arange(len(features)), labels]
    return labels, assigned.astype(np.float32)


def reorder_clusters_by_error_rate(
    labels: np.ndarray,
    speaker_df: pd.DataFrame,
    error_rate_by_speaker: Dict[str, float],
    speaker_col: str = SPEAKER_COL,
) -> Tuple[np.ndarray, List[int]]:
    """Relabel so cluster 0 has the lowest mean train error rate (more native-like)."""
    k = int(labels.max()) + 1 if len(labels) else 0
    rates: List[Tuple[float, int]] = []
    for cluster in range(k):
        members = speaker_df.loc[labels == cluster, speaker_col].astype(str)
        vals = [error_rate_by_speaker.get(s, np.nan) for s in members]
        mean_rate = float(np.nanmean(vals)) if vals else np.nan
        rates.append((mean_rate, cluster))
    order = [old for _, old in sorted(rates, key=lambda item: (np.isnan(item[0]), item[0], item[1]))]
    remap = {old: new for new, old in enumerate(order)}
    new_labels = np.array([remap[int(x)] for x in labels], dtype=np.int64)
    return new_labels, order


def reorder_centroids(centroids: np.ndarray, old_to_new_order: Sequence[int]) -> np.ndarray:
    """`old_to_new_order[new_id] = old_id` from ``reorder_clusters_by_error_rate``."""
    return np.stack([centroids[old] for old in old_to_new_order], axis=0)
