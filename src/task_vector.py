"""Parameter-space correction vectors (Pseudo2Real task arithmetic)."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn

from .asr_backend import load_pretrained_state_dict
from .progress import progress_bar

logger = logging.getLogger(__name__)


def _load_state_dict(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    path = Path(ckpt_dir)
    if not path.exists():
        logger.info("Loading weights from model id %s", ckpt_dir)
        return load_pretrained_state_dict(ckpt_dir)

    candidates = [
        path / "pytorch_model.bin",
        path / "model.safetensors",
        path,
    ]
    # HF trainer may nest under checkpoint-*
    if path.is_dir():
        ckpt_dirs = sorted(path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if ckpt_dirs:
            candidates.insert(0, ckpt_dirs[-1] / "pytorch_model.bin")
            candidates.insert(1, ckpt_dirs[-1] / "model.safetensors")
            candidates.insert(2, ckpt_dirs[-1])

    for cand in candidates:
        if cand.is_file() and cand.suffix == ".bin":
            logger.info("Loading state_dict from %s", cand)
            return torch.load(cand, map_location="cpu")
        if cand.is_file() and cand.suffix == ".safetensors":
            from safetensors.torch import load_file

            logger.info("Loading state_dict from %s", cand)
            return load_file(str(cand), device="cpu")
        if cand.is_dir() and (cand / "config.json").exists():
            logger.info("Loading HF model directory %s", cand)
            return load_pretrained_state_dict(str(cand))

    raise FileNotFoundError(f"Could not locate model weights under {ckpt_dir}")


def calculate_task_vector(
    real_ckpt_dir: str,
    pseudo_ckpt_dir: str,
    save_path: str,
) -> Dict[str, torch.Tensor]:
    """Compute ``tau = theta_real - theta_pseudo`` for matching float tensors."""
    real_sd = _load_state_dict(real_ckpt_dir)
    pseudo_sd = _load_state_dict(pseudo_ckpt_dir)

    shared = sorted(set(real_sd.keys()) & set(pseudo_sd.keys()))
    if not shared:
        raise ValueError("No overlapping parameter keys between real and pseudo checkpoints")

    tau: Dict[str, torch.Tensor] = {}
    skipped = 0
    for key in progress_bar(shared, total=len(shared), desc="compute tau", unit="param"):
        a, b = real_sd[key], pseudo_sd[key]
        if not torch.is_floating_point(a) or not torch.is_floating_point(b):
            skipped += 1
            continue
        if a.shape != b.shape:
            logger.warning("Shape mismatch for %s: %s vs %s; skip", key, a.shape, b.shape)
            skipped += 1
            continue
        tau[key] = (a.float() - b.float()).contiguous()

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tau, out)
    logger.info(
        "Saved correction vector with %d tensors (%d skipped) -> %s",
        len(tau),
        skipped,
        out,
    )
    return tau


def apply_task_vector(
    target_model: nn.Module,
    tau: Dict[str, torch.Tensor],
    lambda_val: float,
    log: bool = True,
) -> nn.Module:
    """In-place: ``param += lambda * tau[key]`` for overlapping float parameters."""
    if abs(lambda_val) < 1e-12:
        return target_model

    applied = 0
    seen_ptrs = set()
    with torch.no_grad():
        state = target_model.state_dict()
        for key, delta in tau.items():
            if key not in state:
                continue
            param = state[key]
            if not torch.is_floating_point(param):
                continue
            if param.shape != delta.shape:
                logger.warning("Skip apply %s due to shape mismatch", key)
                continue
            ptr = int(param.data_ptr())
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            param.add_(delta.to(device=param.device, dtype=param.dtype) * float(lambda_val))
            applied += 1
        target_model.load_state_dict(state, strict=False)
    if log:
        logger.info("Applied lambda=%.3f to %d parameters", lambda_val, applied)
    return target_model


def mix_task_vectors(
    taus: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    """Return ``sum_k weights[k] * tau_k`` over shared float keys."""
    if len(taus) != len(weights):
        raise ValueError(f"Expected {len(taus)} weights, got {len(weights)}")
    mixed: Dict[str, torch.Tensor] = {}
    active = [(tau, float(w)) for tau, w in zip(taus, weights) if abs(float(w)) > 1e-12]
    if not active:
        return mixed
    keys = set(active[0][0].keys())
    for tau, _ in active[1:]:
        keys &= set(tau.keys())
    for key in keys:
        acc: Optional[torch.Tensor] = None
        for tau, weight in active:
            tensor = tau[key]
            if not torch.is_floating_point(tensor):
                acc = None
                break
            term = tensor.float() * weight
            acc = term if acc is None else acc + term
        if acc is not None:
            mixed[key] = acc.contiguous()
    return mixed


def load_tau(tau_path: str) -> Dict[str, torch.Tensor]:
    tau = torch.load(tau_path, map_location="cpu")
    if not isinstance(tau, dict):
        raise TypeError(f"Expected dict tau at {tau_path}, got {type(tau)}")
    return tau


_EMBED_TOKENS = ("embed_tokens", "lm_head", "proj_out")
_QWEN_AUDIO = ("audio_tower", "audio_encoder", "speech_encoder", "audio_model")


def classify_tau_key(key: str, family: str = "whisper") -> str:
    """Map a state_dict key to encoder / decoder / embed / other.

    Whisper: ``model.encoder.*`` vs ``model.decoder.*``; ``embed_tokens`` and
    ``proj_out`` are the text embedding / LM head.
    Qwen3-ASR: audio tower vs causal LM; ``embed_tokens`` / ``lm_head`` are embed.
    """
    name = key[7:] if key.startswith("module.") else key
    lower = name.lower()
    if any(token in lower for token in _EMBED_TOKENS):
        return "embed"
    if str(family).lower() in {"qwen3_asr", "qwen"}:
        if any(token in lower for token in _QWEN_AUDIO):
            return "encoder"
        if ".encoder." in lower and "language" not in lower:
            return "encoder"
        return "decoder"
    if ".encoder." in name or name.startswith("encoder.") or name.startswith("model.encoder"):
        return "encoder"
    if ".decoder." in name or name.startswith("decoder.") or name.startswith("model.decoder"):
        return "decoder"
    return "other"


def filter_tau(
    tau: Dict[str, torch.Tensor],
    groups: Sequence[str],
    *,
    family: str = "whisper",
) -> Dict[str, torch.Tensor]:
    """Keep only parameters whose module group is in ``groups``."""
    wanted = {str(g) for g in groups}
    return {key: tensor for key, tensor in tau.items() if classify_tau_key(key, family=family) in wanted}


def _layer_index_from_key(key: str, side: str) -> Optional[int]:
    """Parse ``...{side}.layers.{i}...``; return None if not a layered weight."""
    marker = f".{side}.layers."
    name = key[7:] if key.startswith("module.") else key
    if marker not in name:
        return None
    rest = name.split(marker, 1)[1]
    idx_str = rest.split(".", 1)[0]
    if not idx_str.isdigit():
        return None
    return int(idx_str)


def list_layer_indices(tau: Dict[str, torch.Tensor], side: str = "encoder") -> List[int]:
    found = set()
    for key in tau:
        idx = _layer_index_from_key(key, side)
        if idx is not None:
            found.add(idx)
    return sorted(found)


def filter_tau_layers(
    tau: Dict[str, torch.Tensor],
    *,
    side: str = "encoder",
    keep_layers: Sequence[int],
    keep_non_layered: bool = False,
) -> Dict[str, torch.Tensor]:
    """Keep only ``side.layers.{i}`` for ``i in keep_layers``.

    Non-layered weights on that side (e.g. conv, layer_norm) are dropped unless
    ``keep_non_layered`` is True. Other module sides are always dropped.
    """
    wanted = {int(i) for i in keep_layers}
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in tau.items():
        name = key[7:] if key.startswith("module.") else key
        if f".{side}." not in name and not name.startswith(f"{side}."):
            continue
        idx = _layer_index_from_key(key, side)
        if idx is None:
            if keep_non_layered:
                out[key] = tensor
            continue
        if idx in wanted:
            out[key] = tensor
    return out


def sparsity_stats(
    original: Dict[str, torch.Tensor],
    sparse: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    total = 0
    kept = 0
    for key, tensor in original.items():
        if not torch.is_floating_point(tensor):
            continue
        n = int(tensor.numel())
        total += n
        other = sparse.get(key)
        if other is None:
            continue
        kept += int((other.float().abs() > 0).sum().item())
    dens = (kept / total) if total else 0.0
    orig_norm = tau_norm_sq(original) ** 0.5
    sparse_norm = tau_norm_sq(sparse) ** 0.5
    return {
        "n_params": float(total),
        "n_nonzero": float(kept),
        "density": float(dens),
        "drop_rate_realized": float(1.0 - dens),
        "l2_original": float(orig_norm),
        "l2_sparse": float(sparse_norm),
        "l2_ratio": float(sparse_norm / orig_norm) if orig_norm > 0 else 0.0,
    }


def dare_sparsify(
    tau: Dict[str, torch.Tensor],
    drop_rate: float,
    *,
    seed: int = 42,
    rescale: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """DARE: randomly drop delta entries and optionally rescale by ``1/(1-p)``.

    Reference: Yu et al., "Language Models are Super Mario" (DARE).
    """
    p = float(drop_rate)
    if p < 0.0 or p >= 1.0:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
    if p == 0.0:
        cloned = {k: v.float().contiguous() for k, v in tau.items() if torch.is_floating_point(v)}
        return cloned, sparsity_stats(tau, cloned)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    scale = 1.0 / (1.0 - p) if rescale else 1.0
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in tau.items():
        if not torch.is_floating_point(tensor):
            continue
        flat = tensor.float().reshape(-1)
        mask = torch.rand(flat.shape, generator=gen) >= p
        sparse = flat * mask.to(flat.dtype)
        if rescale:
            sparse = sparse * scale
        out[key] = sparse.reshape(tensor.shape).contiguous()
    return out, sparsity_stats(tau, out)


def ties_trim(
    tau: Dict[str, torch.Tensor],
    density: float,
    *,
    per_tensor: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """TIES-style magnitude trim for a single task vector.

    Keep the top ``density`` fraction of entries by absolute value; zero the rest.
    With one vector, sign-election is a no-op. ``density`` in (0, 1].
    """
    d = float(density)
    if d <= 0.0 or d > 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    if d >= 1.0 - 1e-12:
        cloned = {k: v.float().contiguous() for k, v in tau.items() if torch.is_floating_point(v)}
        return cloned, sparsity_stats(tau, cloned)

    out: Dict[str, torch.Tensor] = {}
    if per_tensor:
        for key, tensor in tau.items():
            if not torch.is_floating_point(tensor):
                continue
            flat = tensor.float().reshape(-1)
            k = max(1, int(round(flat.numel() * d)))
            if k >= flat.numel():
                out[key] = flat.reshape(tensor.shape).contiguous()
                continue
            thresh = torch.topk(flat.abs(), k, largest=True).values.min()
            mask = flat.abs() >= thresh
            # If ties at threshold inflate density, keep exactly k via topk indices.
            if int(mask.sum().item()) > k:
                idx = torch.topk(flat.abs(), k, largest=True).indices
                mask = torch.zeros_like(flat, dtype=torch.bool)
                mask[idx] = True
            out[key] = (flat * mask.to(flat.dtype)).reshape(tensor.shape).contiguous()
    else:
        flats: List[torch.Tensor] = []
        meta: List[Tuple[str, torch.Size, int]] = []
        for key, tensor in tau.items():
            if not torch.is_floating_point(tensor):
                continue
            flat = tensor.float().reshape(-1)
            meta.append((key, tensor.shape, flat.numel()))
            flats.append(flat)
        if not flats:
            return {}, sparsity_stats(tau, {})
        cat = torch.cat(flats)
        k = max(1, int(round(cat.numel() * d)))
        idx = torch.topk(cat.abs(), k, largest=True).indices
        mask = torch.zeros_like(cat, dtype=torch.bool)
        mask[idx] = True
        sparse_cat = cat * mask.to(cat.dtype)
        offset = 0
        for key, shape, n in meta:
            out[key] = sparse_cat[offset : offset + n].reshape(shape).contiguous()
            offset += n
    return out, sparsity_stats(tau, out)


def save_tau(tau: Dict[str, torch.Tensor], save_path: str) -> Path:
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tau, out)
    logger.info("Saved tau (%d tensors) -> %s", len(tau), out)
    return out


def tau_dot(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, tensor in a.items():
        if key not in b or tensor.shape != b[key].shape:
            continue
        if not torch.is_floating_point(tensor) or not torch.is_floating_point(b[key]):
            continue
        total += float(torch.dot(tensor.float().view(-1), b[key].float().view(-1)))
    return total


def tau_norm_sq(tau: Dict[str, torch.Tensor]) -> float:
    return tau_dot(tau, tau)


def orthogonal_residual(
    tau: Dict[str, torch.Tensor],
    basis: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return ``tau - proj_basis(tau)`` over shared float keys."""
    denom = tau_norm_sq(basis)
    if denom <= 1e-12:
        return {key: tensor.float().contiguous() for key, tensor in tau.items() if torch.is_floating_point(tensor)}
    scale = tau_dot(tau, basis) / denom
    out: Dict[str, torch.Tensor] = {}
    keys = set(tau) | set(basis)
    for key in keys:
        left = tau.get(key)
        right = basis.get(key)
        if left is None or not torch.is_floating_point(left):
            continue
        if right is None or left.shape != right.shape:
            out[key] = left.float().contiguous()
            continue
        out[key] = (left.float() - right.float() * scale).contiguous()
    return out


def tau_gram_matrix(taus: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Return the ``N x N`` Gram matrix ``G_ij = <tau_i, tau_j>``."""
    n = len(taus)
    gram = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        for j in range(i, n):
            value = tau_dot(taus[i], taus[j])
            gram[i, j] = value
            gram[j, i] = value
    return gram


def scale_tau(tau: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    if abs(float(scale) - 1.0) <= 1e-12:
        return {key: tensor.float().contiguous() for key, tensor in tau.items() if torch.is_floating_point(tensor)}
    return mix_task_vectors([tau], [float(scale)])


def svd_subspace_from_taus(
    taus: Sequence[Dict[str, torch.Tensor]],
    rank: int,
) -> Tuple[List[Dict[str, torch.Tensor]], List[float], List[List[float]]]:
    """Low-rank SVD of an N-vector bank via the N×N Gram matrix.

    Returns unit basis ``u_1..u_k``, eigenvalues (descending), and
    loadings ``L[n][j] = <tau_n, u_j>``.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    n = len(taus)
    if n == 0:
        return [], [], []
    gram = tau_gram_matrix(taus)
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)
    k = min(int(rank), n)
    basis: List[Dict[str, torch.Tensor]] = []
    eigenvalues: List[float] = []
    for axis in range(k):
        idx = int(order[axis].item())
        energy = float(evals[idx].item())
        eigenvalues.append(energy)
        if energy <= 1e-12:
            continue
        weights = evecs[:, idx].tolist()
        direction = mix_task_vectors(list(taus), [float(w) for w in weights])
        norm = tau_norm_sq(direction) ** 0.5
        if norm <= 1e-12:
            continue
        basis.append(scale_tau(direction, 1.0 / norm))
    loadings = [[float(tau_dot(tau, axis)) for axis in basis] for tau in taus]
    return basis, eigenvalues, loadings


def project_tau_onto_basis(
    tau: Dict[str, torch.Tensor],
    basis: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Return ``sum_j <tau, u_j> u_j``."""
    acc: Optional[Dict[str, torch.Tensor]] = None
    for axis in basis:
        coeff = tau_dot(tau, axis)
        if abs(coeff) <= 1e-12:
            continue
        term = scale_tau(axis, coeff)
        acc = term if acc is None else mix_task_vectors([acc, term], [1.0, 1.0])
    if acc is None:
        return {key: torch.zeros_like(tensor.float()) for key, tensor in tau.items() if torch.is_floating_point(tensor)}
    return acc


def summarize_tau_modules(tau: Dict[str, torch.Tensor], family: str = "whisper") -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in tau.items():
        grouped.setdefault(classify_tau_key(key, family=family), {})[key] = tensor
    payload: Dict[str, Dict[str, float]] = {}
    full_norm = tau_norm_sq(tau) ** 0.5
    for name, sub in grouped.items():
        nsq = tau_norm_sq(sub)
        payload[name] = {
            "n_tensors": float(len(sub)),
            "l2_norm": nsq ** 0.5,
            "norm_share": (nsq ** 0.5 / full_norm) if full_norm else 0.0,
        }
    return payload


def clone_model_with_tau(
    base_model: nn.Module,
    tau: Dict[str, torch.Tensor],
    lambda_val: float,
) -> nn.Module:
    """Deep-copy ``base_model`` then apply ``lambda * tau`` (non-destructive for base)."""
    model = copy.deepcopy(base_model)
    apply_task_vector(model, tau, lambda_val)
    return model


def grid_search_best_lambda(
    model: nn.Module,
    tau: Dict[str, torch.Tensor],
    dev_loader,
    eval_fn: Callable,
    lambda_list: Optional[Iterable[float]] = None,
    curve_csv_path: Optional[str] = None,
    best_json_path: Optional[str] = None,
) -> Tuple[float, pd.DataFrame]:
    """Search ``lambda`` on Source Dev only (zero target leakage).

    ``eval_fn(model, dataloader) -> dict`` must return at least ``{"wer": float}``.
    """
    if lambda_list is None:
        lambda_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    lambdas = list(lambda_list)

    # Cache original weights so each trial starts from the same base.
    base_state = copy.deepcopy(model.state_dict())
    rows: List[Dict[str, Union[float, int]]] = []

    lam_bar = progress_bar(lambdas, total=len(lambdas), desc="lambda grid search (source_dev)", unit="λ")
    for lam in lam_bar:
        model.load_state_dict(base_state, strict=True)
        apply_task_vector(model, tau, float(lam))
        metrics = eval_fn(model, dev_loader)
        row = {"lambda": float(lam), **{k: float(v) for k, v in metrics.items() if _is_number(v)}}
        rows.append(row)
        wer = row.get("wer", float("nan"))
        lam_bar.set_postfix(wer=f"{wer:.2f}")
        logger.info("lambda=%.2f -> %s", lam, {k: row[k] for k in row if k != "lambda"})

    # Restore original (uncorrected) weights.
    model.load_state_dict(base_state, strict=True)

    curve = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    if "wer" not in curve.columns:
        raise KeyError("eval_fn must return a 'wer' metric for lambda selection")

    best_idx = int(curve["wer"].idxmin())
    best_lambda = float(curve.loc[best_idx, "lambda"])
    best_wer = float(curve.loc[best_idx, "wer"])

    if curve_csv_path:
        Path(curve_csv_path).parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(curve_csv_path, index=False)
        logger.info("Wrote lambda curve to %s", curve_csv_path)

    if best_json_path:
        Path(best_json_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_lambda": best_lambda,
            "best_wer": best_wer,
            "lambda_list": lambdas,
            "selection_set": "source_dev",
            "zero_target_leakage": True,
        }
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote best_lambda=%.2f (WER=%.4f) to %s", best_lambda, best_wer, best_json_path)

    return best_lambda, curve


def _is_number(v: object) -> bool:
    try:
        float(v)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False
