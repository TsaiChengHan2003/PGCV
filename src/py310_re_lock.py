"""Keep Python 3.10's ``re`` compiler from aborting Qwen imports.

CPython 3.10's ``sre`` compiler is not thread-safe, and it also raises
``ValueError: too many values to unpack (expected 0)`` while compiling the
single alternation that ``torch._dynamo.trace_rules`` builds at import.
``transformers`` pulls that in through ``torchvision`` when loading a
processor. Inference does not use Dynamo, so a failed skip-dir regex must
not stop the process.
"""

from __future__ import annotations

import os
import re
import sys
import threading

_LOCK = threading.Lock()
_INSTALLED = False
_WARNED = False
_DYNAMO_PREFIX = "^[^\s<]*("


class _AnyMatch:
    """``re.Pattern`` stand-in: true when any chunk matches."""

    def __init__(self, patterns: list) -> None:
        self._patterns = patterns

    def match(self, text, pos=0, endpos=sys.maxsize):
        for pattern in self._patterns:
            hit = pattern.match(text, pos, endpos)
            if hit:
                return hit
        return None

    def search(self, text, pos=0, endpos=sys.maxsize):
        return self.match(text, pos, endpos)


def _split_alternatives(body: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\" and index + 1 < len(body):
            buf.append(body[index : index + 2])
            index += 2
            continue
        if char == "|":
            parts.append("".join(buf))
            buf = []
            index += 1
            continue
        buf.append(char)
        index += 1
    parts.append("".join(buf))
    return [part for part in parts if part]


def _compile_chunks(original, alternatives: list[str], flags: int, size: int):
    compiled = []
    for start in range(0, len(alternatives), size):
        chunk = alternatives[start : start + size]
        pattern = _DYNAMO_PREFIX + "|".join(chunk) + ")"
        try:
            compiled.append(original(pattern, flags))
        except ValueError as exc:
            if "too many values to unpack" not in str(exc) or len(chunk) == 1:
                continue
            compiled.extend(_compile_chunks(original, chunk, flags, max(1, size // 2)))
    return compiled


def _repair_dynamo_regex(original, pattern: str, flags: int):
    """Compile Dynamo's skip-dir alternation in small pieces."""
    global _WARNED
    if not pattern.startswith(_DYNAMO_PREFIX) or not pattern.endswith(")"):
        raise ValueError("too many values to unpack (expected 0)")
    alternatives = _split_alternatives(pattern[len(_DYNAMO_PREFIX) : -1])
    compiled = _compile_chunks(original, alternatives, flags, 32)
    if not _WARNED:
        _WARNED = True
        sys.stderr.write(
            "py310_re_lock: Python 3.10 could not compile Dynamo's skip-dir regex; "
            f"using {len(compiled)} smaller patterns so the Qwen import can continue.\n"
        )
        sys.stderr.flush()
    if not compiled:
        return original(r"(?!)")
    if len(compiled) == 1:
        return compiled[0]
    return _AnyMatch(compiled)


def install_py310_re_lock() -> None:
    """Install a process-wide compile lock and turn off async weight loading."""
    global _INSTALLED
    # Read before transformers starts its weight-loading pool.
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    if _INSTALLED or getattr(re.compile, "_l2_re_lock", False):
        _INSTALLED = True
        return
    original = re.compile

    def _locked_compile(pattern, flags=0):
        with _LOCK:
            try:
                return original(pattern, flags)
            except ValueError as exc:
                text = pattern if isinstance(pattern, str) else ""
                if "too many values to unpack" not in str(exc) or not text.startswith(_DYNAMO_PREFIX):
                    raise
                return _repair_dynamo_regex(original, text, flags)

    _locked_compile._l2_re_lock = True  # type: ignore[attr-defined]
    re.compile = _locked_compile
    _INSTALLED = True
