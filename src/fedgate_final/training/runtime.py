"""Runtime settings for faster CUDA execution."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def apply_runtime_settings(runtime_cfg: Mapping[str, Any] | None) -> None:
    cfg = dict(runtime_cfg or {})
    if not cfg:
        return

    if torch.cuda.is_available():
        allow_tf32 = cfg.get("allow_tf32")
        if allow_tf32 is not None:
            value = bool(allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = value
            torch.backends.cudnn.allow_tf32 = value

        cudnn_benchmark = cfg.get("cudnn_benchmark")
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    precision = cfg.get("float32_matmul_precision")
    if precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(precision))


def resolve_mixed_precision_dtype(
    runtime_cfg: Mapping[str, Any] | None,
    *,
    device: torch.device,
) -> torch.dtype | None:
    if device.type != "cuda":
        return None

    cfg = dict(runtime_cfg or {})
    value = str(cfg.get("mixed_precision", "")).strip().lower()
    if not value or value in {"0", "false", "no", "off", "none"}:
        return None
    if value in {"auto", "bf16", "bfloat16"}:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if value == "auto":
            return torch.float16
        return None
    if value in {"fp16", "float16", "half", "16"}:
        return torch.float16
    return None
