"""Experiment progress bars (tqdm) for academic ASR / ML runs.

Design goals (common in paper reproduction / lab scripts):
  - Always-visible bars even when stdout is piped / IDE-captured (force enable)
  - Logging goes through ``tqdm.write`` so bars are not corrupted
  - Standard bar format: ``desc |████| n/N [elapsed<remain, rate]``
  - Pipeline-level stage banners + outer stage progress
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional, TypeVar

from tqdm.auto import tqdm as _tqdm_auto
from tqdm import tqdm as _tqdm_std

T = TypeVar("T")

# Module-level switch (set by configure_experiment_progress).
_FORCE_ENABLE = True
_DISABLE = False

# Academic-style bar: percent + counts + ETA + throughput.
BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


class TqdmLoggingHandler(logging.Handler):
    """Route log records through ``tqdm.write`` to avoid breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _tqdm_std.write(msg, file=sys.stderr)
            self.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)


def configure_experiment_progress(
    *,
    force: bool = True,
    disable: bool = False,
    mininterval: float = 0.3,
) -> None:
    """Call once at process start (e.g. in ``run_pipeline.main``).

    Parameters
    ----------
    force:
        Force-show bars even when not attached to a TTY (Cursor / nohup / tee).
    disable:
        Globally turn off all progress bars (e.g. ``--no-progress``).
    mininterval:
        Minimum refresh interval in seconds (reduces flicker on slow terminals).
    """
    global _FORCE_ENABLE, _DISABLE
    _FORCE_ENABLE = force and not disable
    _DISABLE = disable

    # Make nested HF / third-party tqdm respect the same preference when possible.
    if disable:
        os.environ["TQDM_DISABLE"] = "1"
    else:
        os.environ.pop("TQDM_DISABLE", None)
        os.environ["TQDM_MININTERVAL"] = str(mininterval)

    # Re-wire root logging handlers to be tqdm-safe (idempotent-ish).
    root = logging.getLogger()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    tqdm_handler_present = any(isinstance(h, TqdmLoggingHandler) for h in root.handlers)
    if not tqdm_handler_present:
        # Replace StreamHandlers that fight with the bar.
        kept = []
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TqdmLoggingHandler):
                root.removeHandler(h)
            else:
                kept.append(h)
        handler = TqdmLoggingHandler()
        handler.setFormatter(fmt)
        handler.setLevel(logging.INFO)
        root.addHandler(handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)


def progress_bar(
    iterable: Optional[Iterable[T]] = None,
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    position: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Drop-in tqdm wrapper used across the codebase."""
    disable = _DISABLE or kwargs.pop("disable", False)
    if not disable and _FORCE_ENABLE:
        # Explicit False overrides tqdm's auto-TTY detection.
        disable = False
    elif not _FORCE_ENABLE and not disable:
        disable = None  # tqdm default: hide when not a TTY

    return _tqdm_auto(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        position=position,
        disable=disable,
        dynamic_ncols=True,
        mininterval=float(os.environ.get("TQDM_MININTERVAL", "0.3")),
        bar_format=kwargs.pop("bar_format", BAR_FORMAT),
        file=kwargs.pop("file", sys.stderr),
        **kwargs,
    )


def stage_banner(stage_id: int, title: str) -> None:
    """Print a clear stage separator (common in multi-stage experiment scripts)."""
    line = "=" * 72
    msg = f"\n{line}\n[Stage {stage_id}] {title}\n{line}"
    _tqdm_std.write(msg, file=sys.stderr)


@contextmanager
def stage_progress(stages: Iterable[int], desc: str = "pipeline stages") -> Iterator[Any]:
    """Outer progress over selected pipeline stages (advances when a stage finishes)."""
    ordered = sorted(set(stages))
    pbar = progress_bar(total=len(ordered), desc=desc, unit="stage", leave=True)
    done: set[int] = set()

    class _Ctrl:
        def begin(self, stage_id: int, title: str = "") -> None:
            stage_banner(stage_id, title or f"stage {stage_id}")
            pbar.set_postfix_str(f"running={stage_id}")

        def end(self, stage_id: int) -> None:
            if stage_id in ordered and stage_id not in done:
                done.add(stage_id)
                pbar.update(1)
                pbar.set_postfix_str(f"done={stage_id}")

        # Back-compat alias used as begin+immediate end if needed.
        def tick(self, stage_id: int, title: str = "") -> None:
            self.begin(stage_id, title)

    try:
        yield _Ctrl()
    finally:
        # Fill any remaining if caller forgot end(); keep bar honest.
        remaining = len(ordered) - len(done)
        if remaining > 0:
            pbar.update(remaining)
        pbar.close()


def hf_trainer_tqdm_disabled() -> bool:
    """Value for ``TrainingArguments.disable_tqdm``."""
    return bool(_DISABLE)
