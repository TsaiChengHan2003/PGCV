#!/usr/bin/env python3
"""Per-level full fine-tune on TEEMI A01, scored on that level only.

Stages:
1. Slice existing A01 train/dev/test and A02 CSVs by proficiency_level.
2. Full fine-tune one model per level on that level's gold train text.
3. Decode pretrained and the matched checkpoint on that level's eval slices.
4. Write the specialized-WER report.

Gold CEFR only chooses the training bucket and the matched eval bucket.
There is no task vector and no cross-level decode.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr_backend import family_slug, load_asr_model, load_processor as load_asr_processor  # noqa: E402
from src.audio_utils import ensure_device  # noqa: E402
from src.evaluate import build_eval_dataloader, evaluate_model  # noqa: E402
from src.finetune import FinetuneConfig, finetune_whisper  # noqa: E402
from src.progress import configure_experiment_progress, progress_bar, stage_progress  # noqa: E402

logger = logging.getLogger("teemi_a01_level_full_ft")

PRETRAINED = "pretrained"
LEVEL_FT = "level_full_ft"
SYSTEMS = (PRETRAINED, LEVEL_FT)


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict config, got {type(cfg)}")
    return cfg


def parse_stages(raw: str) -> Set[int]:
    aliases = {
        "prepare": 1,
        "1": 1,
        "train": 2,
        "2": 2,
        "eval": 3,
        "3": 3,
        "report": 4,
        "4": 4,
    }
    text = str(raw).strip().lower()
    if text == "all":
        return {1, 2, 3, 4}
    stages: Set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in aliases:
            raise ValueError(
                f"Unsupported stage {part!r}; choose prepare, train, eval, report, 1-4, or all"
            )
        stages.add(aliases[part])
    if not stages:
        raise ValueError("No stages selected")
    return stages


def stage_titles() -> Dict[int, str]:
    return {
        1: "Materialize per-level gold splits",
        2: "Full fine-tune one model per level",
        3: "Score specialized WER on the matched level",
        4: "Write experiment report",
    }


def family_root(cfg: Dict[str, Any], key: str) -> Path:
    return resolve_repo_path(cfg[key]) / family_slug(cfg["student_model"])


def levels_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    return [str(x) for x in cfg["levels"]]


def selected_levels(cfg: Dict[str, Any], requested: Sequence[str]) -> List[str]:
    allowed = levels_from_cfg(cfg)
    if not requested:
        return allowed
    wanted: List[str] = []
    for item in requested:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                wanted.append(part)
    unknown = [level for level in wanted if level not in allowed]
    if unknown:
        raise ValueError(f"Unknown level(s) {unknown}; choose from {allowed}")
    chosen = set(wanted)
    return [level for level in allowed if level in chosen]


def slugify_level(level: str) -> str:
    text = level.lower().replace("+", "_plus")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def data_root(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "data_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "results_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def ckpt_root(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "checkpoint_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_csv(cfg: Dict[str, Any], split_key: str) -> Path:
    return resolve_repo_path(cfg["splits"][split_key])


def level_split_csv(cfg: Dict[str, Any], split_key: str, level: str) -> Path:
    return data_root(cfg) / "levels" / slugify_level(level) / f"{split_key}.csv"


def level_ckpt_dir(cfg: Dict[str, Any], level: str) -> Path:
    return ckpt_root(cfg) / slugify_level(level)


def metrics_path(cfg: Dict[str, Any], split_key: str, level: str, system: str) -> Path:
    return results_dir(cfg) / "by_level" / slugify_level(level) / split_key / system / "metrics.json"


def predictions_path(cfg: Dict[str, Any], split_key: str, level: str, system: str) -> Path:
    return results_dir(cfg) / "by_level" / slugify_level(level) / split_key / system / "predictions.csv"


def eval_splits_from_cfg(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in cfg.get("eval_splits", []):
        pairs.append((str(item["key"]), str(item.get("label", item["key"]))))
    return pairs


def min_train_utterances(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("min_train_utterances", 16))


def decode_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "language": str(cfg.get("language", "en")),
        "task": str(cfg.get("task", "transcribe")),
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
    }


def row_count(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    return len(pd.read_csv(csv_path))


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.is_dir():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def step_of(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    return max(checkpoints, key=step_of)


def _checkpoint_global_step(checkpoint_dir: Path) -> Optional[int]:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        return None
    step = json.loads(state_path.read_text(encoding="utf-8")).get("global_step")
    return int(step) if step is not None else None


def _has_weights(output_dir: Path) -> bool:
    return (
        (output_dir / "model.safetensors").is_file()
        or (output_dir / "pytorch_model.bin").is_file()
        or (output_dir / "model.safetensors.index.json").is_file()
    )


def training_completed(output_dir: Path, max_steps: int) -> bool:
    if _has_weights(output_dir):
        latest = _latest_checkpoint(output_dir)
        if latest is None:
            return True
        step = _checkpoint_global_step(latest)
        return step is None or step >= max_steps
    latest = _latest_checkpoint(output_dir)
    if latest is None:
        return False
    step = _checkpoint_global_step(latest)
    return step is not None and step >= max_steps


def resolve_weight_dir(output_dir: Path) -> Path:
    if _has_weights(output_dir):
        return output_dir
    latest = _latest_checkpoint(output_dir)
    if latest is not None:
        return latest
    if (output_dir / "config.json").is_file():
        return output_dir
    raise FileNotFoundError(f"Could not locate evaluable weights under {output_dir}")


def _slice_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if "proficiency_level" not in df.columns:
        raise KeyError("Missing proficiency_level column")
    part = df[df["proficiency_level"].astype(str).str.strip() == str(level)].copy()
    sort_cols = [c for c in ["booklet", "global_speaker_id", "utt_id"] if c in part.columns]
    if sort_cols and len(part):
        part = part.sort_values(sort_cols)
    return part


def materialize_level_splits(cfg: Dict[str, Any]) -> None:
    frames = {split: pd.read_csv(split_csv(cfg, split)) for split in cfg["splits"]}
    rows: List[Dict[str, Any]] = []
    for level in levels_from_cfg(cfg):
        rec: Dict[str, Any] = {"level": level}
        for split_name, df in frames.items():
            part = _slice_level(df, level)
            dest = level_split_csv(cfg, split_name, level)
            dest.parent.mkdir(parents=True, exist_ok=True)
            part.to_csv(dest, index=False)
            rec[f"{split_name}_utterances"] = int(len(part))
            if "global_speaker_id" in part.columns:
                rec[f"{split_name}_speakers"] = int(part["global_speaker_id"].nunique())
        rows.append(rec)
        logger.info(
            "Level %s train %d utt / %d speakers",
            level,
            rec.get("train_utterances", 0),
            rec.get("train_speakers", 0),
        )
    manifest = pd.DataFrame(rows)
    out_csv = results_dir(cfg) / "level_full_ft_manifest.csv"
    manifest.to_csv(out_csv, index=False)
    lines = [
        f"# {cfg.get('display_name', 'Per-level full FT')} Manifest",
        "",
        "Each row is the gold-text slice of one `proficiency_level`.",
        "Training uses that level's A01 train utterances. Scoring uses the same level only.",
        "",
        "| " + " | ".join(manifest.columns) + " |",
        "|" + "|".join("---" if c == "level" else "---:" for c in manifest.columns) + "|",
    ]
    for rec in rows:
        vals = [str(rec[col]) if col == "level" else str(int(rec[col])) for col in manifest.columns]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    (results_dir(cfg) / "level_full_ft_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    write_pipeline_flow(cfg)
    logger.info("[prepare] Wrote %s", out_csv)


def write_pipeline_flow(cfg: Dict[str, Any]) -> None:
    levels = ", ".join(levels_from_cfg(cfg))
    eval_labels = ", ".join(label for _, label in eval_splits_from_cfg(cfg))
    lines = [
        "# 分級 full fine-tune：資料流說明",
        "",
        "每個 CEFR 等級各自做一次 **gold-text full fine-tune**，",
        "然後只在同一個等級的語句上計算 WER。",
        "",
        "這不是 error-GT 子集，也不做 task vector 或跨等級套用。",
        "",
        "## 設定",
        "",
        f"- Backbone：`{cfg['student_model']}`",
        f"- 等級：{levels}",
        f"- 訓練：A01 train 中 `proficiency_level == L` 的全部金標句",
        f"- 專項 WER：{eval_labels} 中同一個 L 的語句",
        f"- 對照：同一個切片上的 pretrained",
        f"- 少於 {min_train_utterances(cfg)} 句的等級不訓練",
        "",
        "## Stage",
        "",
        "```",
        "既有 A01 / A02 CSV",
        "    ├─[prepare] 依 proficiency_level 切開",
        "    ├─[train]   每個等級 full fine-tune",
        "    ├─[eval]    pretrained 與該等級權重，只解該等級",
        "    └─[report]  專項 WER 表",
        "```",
        "",
        "## 輸出",
        "",
        f"`{cfg.get('results_dir')}/<model_family>/`",
        "",
        "- `level_full_ft_manifest.md`",
        "- `by_level/<level>/<split>/{pretrained,level_full_ft}/`",
        "- `level_full_ft_wer.csv`",
        "- `experiment_report.md`",
        "",
    ]
    path = results_dir(cfg) / "pipeline_flow.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def train_level_models(cfg: Dict[str, Any], levels: Sequence[str]) -> None:
    max_steps = int(cfg.get("max_steps", 4000))
    floor = min_train_utterances(cfg)
    for level in progress_bar(list(levels), total=len(levels), desc="level full FT", unit="L"):
        csv_path = level_split_csv(cfg, "train", level)
        n = row_count(csv_path)
        if n < floor:
            logger.warning("Skip train %s; only %d train rows (min %d)", level, n, floor)
            continue
        out_dir = level_ckpt_dir(cfg, level)
        if training_completed(out_dir, max_steps):
            logger.info("Skip train %s; completed at %s", level, out_dir)
            continue
        resume = _latest_checkpoint(out_dir)
        dev_csv = level_split_csv(cfg, "dev", level)
        logger.info("Full fine-tune %s (%d utt) -> %s", level, n, out_dir)
        finetune_whisper(
            FinetuneConfig(
                model_name_or_path=str(cfg["student_model"]),
                train_csv=str(csv_path),
                output_dir=str(out_dir),
                text_column="text",
                language=str(cfg.get("language", "en")),
                task=str(cfg.get("task", "transcribe")),
                learning_rate=float(cfg.get("learning_rate", 1e-5)),
                weight_decay=float(cfg.get("weight_decay", 0.01)),
                warmup_steps=int(cfg.get("warmup_steps", 500)),
                max_steps=max_steps,
                per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 12)),
                per_device_eval_batch_size=int(cfg.get("eval_batch_size", 4)),
                gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
                fp16=bool(cfg.get("fp16", True)),
                bf16=bool(cfg.get("bf16", False)),
                seed=int(cfg.get("seed", 42)),
                dataloader_num_workers=int(cfg.get("num_workers", 4)),
                resume_from_checkpoint=str(resume) if resume else None,
                eval_csv=str(dev_csv) if dev_csv.is_file() and row_count(dev_csv) > 0 else None,
                eval_steps=int(cfg.get("eval_steps", 500)),
                save_steps=int(cfg.get("save_steps", 1000)),
                save_total_limit=int(cfg.get("save_total_limit", 2)),
                load_best_model_at_end=bool(cfg.get("load_best_model_at_end", False)),
                metric_for_best_model=str(cfg.get("metric_for_best_model", "eval_loss")),
            )
        )


def _decode_csv(cfg: Dict[str, Any], model, processor, csv_path: Path, dest: Path, desc: str) -> Dict[str, float]:
    loader = build_eval_dataloader(
        str(csv_path),
        processor=processor,
        batch_size=int(cfg.get("eval_batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 2)),
        language=str(cfg.get("language", "en")),
    )
    return evaluate_model(model, loader, processor, output_csv_path=str(dest), desc=desc, **decode_kwargs(cfg))


def _release(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_levels(cfg: Dict[str, Any], levels: Sequence[str], device: str) -> None:
    device_t = ensure_device(device)
    processor = load_asr_processor(cfg["student_model"])
    pending_pretrained: List[Tuple[str, str, Path]] = []
    pending_ft: List[Tuple[str, str, Path]] = []
    for level in levels:
        for split_key, _label in eval_splits_from_cfg(cfg):
            csv_path = level_split_csv(cfg, split_key, level)
            n = row_count(csv_path)
            if n == 0:
                logger.warning("Skip empty slice %s / %s", level, split_key)
                continue
            if not metrics_path(cfg, split_key, level, PRETRAINED).is_file():
                pending_pretrained.append((level, split_key, csv_path))
            if not metrics_path(cfg, split_key, level, LEVEL_FT).is_file():
                pending_ft.append((level, split_key, csv_path))

    if pending_pretrained:
        model = load_asr_model(str(cfg.get("base_model", cfg["student_model"])), device=device_t)
        for level, split_key, csv_path in progress_bar(
            pending_pretrained, total=len(pending_pretrained), desc="pretrained specialized WER", unit="slice"
        ):
            dest = predictions_path(cfg, split_key, level, PRETRAINED)
            metrics = _decode_csv(cfg, model, processor, csv_path, dest, f"pretrained {level} {split_key}")
            dump_json(
                metrics_path(cfg, split_key, level, PRETRAINED),
                {"level": level, "split": split_key, "system": PRETRAINED, "n_csv": row_count(csv_path), **metrics},
            )
        _release(model)
    else:
        logger.info("Pretrained specialized WER already scored for the requested levels")

    for level in levels:
        jobs = [item for item in pending_ft if item[0] == level]
        if not jobs:
            continue
        out_dir = level_ckpt_dir(cfg, level)
        if not training_completed(out_dir, int(cfg.get("max_steps", 4000))) and not _has_weights(out_dir):
            logger.warning("Skip eval %s; no completed checkpoint at %s", level, out_dir)
            continue
        weight_dir = resolve_weight_dir(out_dir)
        model = load_asr_model(str(weight_dir), device=device_t)
        for _level, split_key, csv_path in progress_bar(
            jobs, total=len(jobs), desc=f"{level} full FT", unit="slice"
        ):
            dest = predictions_path(cfg, split_key, level, LEVEL_FT)
            metrics = _decode_csv(cfg, model, processor, csv_path, dest, f"level_full_ft {level} {split_key}")
            dump_json(
                metrics_path(cfg, split_key, level, LEVEL_FT),
                {
                    "level": level,
                    "split": split_key,
                    "system": LEVEL_FT,
                    "checkpoint": str(weight_dir),
                    "n_csv": row_count(csv_path),
                    **metrics,
                },
            )
        _release(model)


def _collect_rows(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = results_dir(cfg) / "by_level"
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("*/*/*/metrics.json")):
        payload = load_json(path)
        rows.append(
            {
                "level": payload.get("level", ""),
                "split": payload.get("split", ""),
                "system": payload.get("system", ""),
                "wer": payload.get("wer"),
                "sub": payload.get("sub"),
                "del": payload.get("del"),
                "ins": payload.get("ins"),
                "n_utterances": payload.get("n_utterances"),
                "n_ref_words": payload.get("n_ref_words"),
                "checkpoint": payload.get("checkpoint", ""),
            }
        )
    return rows


def write_report(cfg: Dict[str, Any]) -> None:
    rows = _collect_rows(cfg)
    out_dir = results_dir(cfg)
    table = pd.DataFrame(rows)
    if len(table):
        table.to_csv(out_dir / "level_full_ft_wer.csv", index=False)
    else:
        logger.warning("No metrics yet; report will only describe the setup")

    eval_splits = eval_splits_from_cfg(cfg)
    lines = [
        f"# {cfg.get('display_name', 'Per-level full FT')}",
        "",
        "## Scope",
        "",
        f"- Backbone: `{cfg['student_model']}`.",
        "- Each level is a separate gold-text full fine-tune on A01 train utterances of that `proficiency_level`.",
        "- Specialized WER is utterance WER on the same level only. Gold CEFR is not a test-time input.",
        f"- Recipe: lr={cfg.get('learning_rate')}, max_steps={cfg.get('max_steps')}, "
        f"batch={cfg.get('per_device_train_batch_size')}×{cfg.get('gradient_accumulation_steps', 1)}.",
        "",
        "## Specialized WER",
        "",
        "Δ is pretrained WER minus level full-FT WER. Positive Δ means the level checkpoint is better.",
        "",
    ]
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        lookup[(str(row["level"]), str(row["split"]), str(row["system"]))] = row

    for split_key, label in eval_splits:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Level | N utt | Pretrained | Level full FT | Δ |")
        lines.append("|---|---:|---:|---:|---:|")
        for level in levels_from_cfg(cfg):
            pre = lookup.get((level, split_key, PRETRAINED))
            ft = lookup.get((level, split_key, LEVEL_FT))
            if pre is None and ft is None:
                continue
            n = ""
            if pre and pre.get("n_utterances") is not None:
                n = str(int(pre["n_utterances"]))
            elif ft and ft.get("n_utterances") is not None:
                n = str(int(ft["n_utterances"]))
            pre_wer = f"{float(pre['wer']):.2f}" if pre and pre.get("wer") is not None else "—"
            ft_wer = f"{float(ft['wer']):.2f}" if ft and ft.get("wer") is not None else "—"
            if pre and ft and pre.get("wer") is not None and ft.get("wer") is not None:
                delta = f"{float(pre['wer']) - float(ft['wer']):+.2f}"
            else:
                delta = "—"
            lines.append(f"| {level} | {n} | {pre_wer} | {ft_wer} | {delta} |")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- A missing level full-FT cell means that checkpoint has not been trained or scored yet.",
            "- A01 test and A02 stay out of training. Dev is recorded, and is also the in-training eval split when it is non-empty.",
            "",
        ]
    )
    report = out_dir / "experiment_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="TEEMI per-level full fine-tune and specialized WER")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--level",
        action="append",
        default=[],
        help="Optional proficiency level. Repeat or comma-separate. Default: every level in the config.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    configure_experiment_progress(force=not args.no_progress, disable=bool(args.no_progress))

    cfg = load_config(args.config)
    stages = parse_stages(args.stage)
    levels = selected_levels(cfg, args.level)
    titles = stage_titles()
    logger.info(
        "Config=%s stages=%s levels=%s device=%s",
        args.config,
        sorted(stages),
        levels,
        args.device,
    )

    with stage_progress(sorted(stages), desc=cfg.get("display_name", "Per-level full FT")) as sp:
        if 1 in stages:
            sp.begin(1, titles[1])
            materialize_level_splits(cfg)
            sp.end(1)
        if 2 in stages:
            sp.begin(2, titles[2])
            train_level_models(cfg, levels)
            sp.end(2)
        if 3 in stages:
            sp.begin(3, titles[3])
            evaluate_levels(cfg, levels, device=args.device)
            sp.end(3)
        if 4 in stages:
            sp.begin(4, titles[4])
            write_report(cfg)
            sp.end(4)


if __name__ == "__main__":
    main()
