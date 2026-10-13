"""Learned speaker-level gates over a frozen prototype-tau bank.

Cross-attention matches a speaker embedding (query) to prototype keys.
The MLP ablation uses the same loss but no prototype keys.

Supervision is WER-oracle routing: which frozen τ_k actually lowers that
speaker's WER. Gold CEFR is not an input.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .progress import progress_bar

logger = logging.getLogger(__name__)

SPEAKER_COL = "global_speaker_id"


class PrototypeCrossAttnGate(nn.Module):
    """Query = speaker embedding; keys = L2 prototype bank.

    ``score_k = (W_q x) · (W_k p_k) / sqrt(d)``
    ``p_k`` is initialized from the mean embedding of train speakers whose
    WER-oracle label is k, then optionally fine-tuned.
    """

    def __init__(
        self,
        in_dim: int,
        n_prototypes: int,
        d_model: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_prototypes = int(n_prototypes)
        self.d_model = int(d_model)
        self.query = nn.Linear(self.in_dim, self.d_model, bias=False)
        self.key = nn.Linear(self.in_dim, self.d_model, bias=False)
        self.proto = nn.Parameter(torch.zeros(self.n_prototypes, self.in_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.scale = self.d_model**0.5

    def init_prototypes(self, means: np.ndarray) -> None:
        matrix = torch.as_tensor(means, dtype=torch.float32)
        if matrix.shape != (self.n_prototypes, self.in_dim):
            raise ValueError(
                f"Prototype means shape {tuple(matrix.shape)} != "
                f"({self.n_prototypes}, {self.in_dim})"
            )
        with torch.no_grad():
            self.proto.copy_(matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.dropout(self.query(x))
        k = self.key(self.proto)
        return (q @ k.t()) / self.scale


class MLPGate(nn.Module):
    """Architecture ablation: no prototype keys, same routing loss."""

    def __init__(
        self,
        in_dim: int,
        n_prototypes: int,
        d_model: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), int(n_prototypes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_gate(
    kind: str,
    in_dim: int,
    n_prototypes: int,
    d_model: int = 64,
    dropout: float = 0.3,
) -> nn.Module:
    name = str(kind).strip().lower()
    if name in {"crossattn", "cross_attn", "prototype_crossattn"}:
        return PrototypeCrossAttnGate(in_dim, n_prototypes, d_model=d_model, dropout=dropout)
    if name in {"mlp", "mlp_gate"}:
        return MLPGate(in_dim, n_prototypes, d_model=d_model, dropout=dropout)
    raise ValueError(f"Unknown gate kind {kind!r}")


def soft_targets_from_wer(wer_matrix: np.ndarray, temperature: float) -> np.ndarray:
    """``softmax(-WER / T)`` over prototypes. ``wer_matrix`` is [N, K] fractions."""
    logits = -np.asarray(wer_matrix, dtype=np.float64) / max(float(temperature), 1e-8)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def prototype_means_from_oracle(
    embeddings: np.ndarray,
    hard_labels: np.ndarray,
    n_prototypes: int,
) -> np.ndarray:
    """Mean L2-normalized embedding per oracle class; empty classes get the global mean."""
    dim = embeddings.shape[1]
    means = np.zeros((n_prototypes, dim), dtype=np.float32)
    global_mean = embeddings.mean(axis=0)
    for idx in range(n_prototypes):
        mask = hard_labels == idx
        vec = embeddings[mask].mean(axis=0) if np.any(mask) else global_mean
        norm = np.linalg.norm(vec)
        means[idx] = (vec / max(float(norm), 1e-8)).astype(np.float32)
    return means


def _kl_to_soft(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, targets, reduction="batchmean")


@torch.no_grad()
def gate_predictions(
    model: nn.Module,
    embeddings: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (alpha [N,K], hard labels [N])."""
    model.eval()
    resolved = device or next(model.parameters()).device
    x = torch.as_tensor(embeddings, dtype=torch.float32)
    alphas: List[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        chunk = x[start : start + batch_size].to(resolved)
        alpha = torch.softmax(model(chunk), dim=-1)
        alphas.append(alpha.cpu().numpy())
    alpha_np = np.concatenate(alphas, axis=0) if alphas else np.zeros((0, 0), dtype=np.float32)
    hard = alpha_np.argmax(axis=1).astype(np.int64) if len(alpha_np) else np.zeros((0,), dtype=np.int64)
    return alpha_np.astype(np.float32), hard


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    if len(target) == 0:
        return float("nan")
    return float((np.asarray(pred) == np.asarray(target)).mean())


def train_gate(
    model: nn.Module,
    train_x: np.ndarray,
    train_soft: np.ndarray,
    *,
    val_x: Optional[np.ndarray] = None,
    val_soft: Optional[np.ndarray] = None,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    patience: int = 20,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train with KL(softmax(logits) || WER-oracle soft labels). Early-stop on val KL."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    resolved = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved)
    dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_soft, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []
    stalled = 0

    epoch_bar = progress_bar(range(int(epochs)), total=int(epochs), desc="train gate", unit="ep")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(resolved)
            yb = yb.to(resolved)
            loss = _kl_to_soft(model(xb), yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)
        train_kl = total_loss / max(n_seen, 1)
        train_alpha, train_hard = gate_predictions(model, train_x, device=resolved)
        train_oracle = train_soft.argmax(axis=1)
        row = {
            "epoch": float(epoch + 1),
            "train_kl": train_kl,
            "train_oracle_acc": accuracy(train_hard, train_oracle),
        }
        val_kl = train_kl
        if val_x is not None and val_soft is not None and len(val_x):
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.as_tensor(val_x, dtype=torch.float32, device=resolved))
                val_kl = float(_kl_to_soft(val_logits, torch.as_tensor(val_soft, dtype=torch.float32, device=resolved)).item())
            val_alpha, val_hard = gate_predictions(model, val_x, device=resolved)
            row["val_kl"] = val_kl
            row["val_oracle_acc"] = accuracy(val_hard, val_soft.argmax(axis=1))
        history.append(row)
        epoch_bar.set_postfix(kl=f"{train_kl:.4f}", val=f"{val_kl:.4f}")

        if val_kl < best_val - 1e-6:
            best_val = val_kl
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
            if stalled >= int(patience):
                logger.info("Early stop at epoch %d (best %d, val_kl=%.4f)", epoch + 1, best_epoch, best_val)
                break

    model.load_state_dict(best_state, strict=True)
    model.to(resolved)
    return {
        "best_epoch": best_epoch,
        "best_val_kl": best_val,
        "history": history,
        "train_oracle_acc": history[best_epoch - 1]["train_oracle_acc"] if history else float("nan"),
    }


def train_gate_ce(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    val_x: Optional[np.ndarray] = None,
    val_y: Optional[np.ndarray] = None,
    class_weights: Optional[np.ndarray] = None,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    patience: int = 20,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train a gate with train-speaker CEFR labels and early-stop on held-out CE."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    resolved = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved)
    dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    weights = (
        torch.as_tensor(class_weights, dtype=torch.float32, device=resolved)
        if class_weights is not None
        else None
    )

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []
    stalled = 0

    epoch_bar = progress_bar(range(int(epochs)), total=int(epochs), desc="train CEFR gate", unit="ep")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(resolved)
            yb = yb.to(resolved)
            loss = F.cross_entropy(model(xb), yb, weight=weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)

        train_ce = total_loss / max(n_seen, 1)
        _, train_hard = gate_predictions(model, train_x, device=resolved)
        row = {
            "epoch": float(epoch + 1),
            "train_ce": train_ce,
            "train_cefr_acc": accuracy(train_hard, train_y),
        }
        val_ce = train_ce
        if val_x is not None and val_y is not None and len(val_x):
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.as_tensor(val_x, dtype=torch.float32, device=resolved))
                val_target = torch.as_tensor(val_y, dtype=torch.long, device=resolved)
                val_ce = float(F.cross_entropy(val_logits, val_target, weight=weights).item())
            _, val_hard = gate_predictions(model, val_x, device=resolved)
            row["val_ce"] = val_ce
            row["val_cefr_acc"] = accuracy(val_hard, val_y)
        history.append(row)
        epoch_bar.set_postfix(ce=f"{train_ce:.4f}", val=f"{val_ce:.4f}")

        if val_ce < best_val - 1e-6:
            best_val = val_ce
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
            if stalled >= int(patience):
                logger.info(
                    "Early stop at epoch %d (best %d, val_ce=%.4f)",
                    epoch + 1,
                    best_epoch,
                    best_val,
                )
                break

    model.load_state_dict(best_state, strict=True)
    model.to(resolved)
    best_row = history[best_epoch - 1] if history and best_epoch else {}
    return {
        "best_epoch": best_epoch,
        "best_val_ce": best_val,
        "history": history,
        "train_cefr_acc": best_row.get("train_cefr_acc", float("nan")),
        "val_cefr_acc": best_row.get("val_cefr_acc", float("nan")),
    }


def save_gate(model: nn.Module, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "class_name": type(model).__name__,
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info("Saved gate -> %s", path)


def load_gate(path: Path, model: nn.Module) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    extra = payload.get("extra") or {}
    logger.info("Loaded gate %s (%s)", path, payload.get("class_name"))
    return extra


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def assignment_frame(
    speakers: Sequence[str],
    hard: np.ndarray,
    alpha: np.ndarray,
    prototype_names: Sequence[str],
    proficiency: Optional[Sequence[str]] = None,
    n_utterances: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, speaker in enumerate(speakers):
        k = int(hard[i])
        rec: Dict[str, Any] = {
            SPEAKER_COL: str(speaker),
            "prototype_id": k,
            "prototype_name": str(prototype_names[k]),
            "max_alpha": float(alpha[i, k]),
        }
        if n_utterances is not None:
            rec["n_utterances"] = int(n_utterances[i])
        if proficiency is not None:
            rec["proficiency_level"] = str(proficiency[i])
        for j, name in enumerate(prototype_names):
            rec[f"alpha_{name}"] = float(alpha[i, j])
        rows.append(rec)
    return pd.DataFrame(rows)
