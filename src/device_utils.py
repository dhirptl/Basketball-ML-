"""Resolve best available torch device."""

from __future__ import annotations

import torch


def resolve_device(explicit: str | None = None) -> str:
    if explicit and explicit.lower() not in ("auto", ""):
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
