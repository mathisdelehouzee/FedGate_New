"""Progress helpers with an optional tqdm dependency."""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Iterator


try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    _tqdm = None


class _PlainProgress:
    def __init__(self, iterable: Iterable[Any], total: int | None = None, desc: str = "") -> None:
        self._iterable = iterable
        self.total = total
        self.desc = desc

    def __iter__(self) -> Iterator[Any]:
        return iter(self._iterable)

    def set_postfix(self, **kwargs: Any) -> None:
        del kwargs

    def write(self, message: str) -> None:
        print(message, flush=True)

    def close(self) -> None:
        return None


def _should_use_tqdm() -> bool:
    if _tqdm is None:
        return False
    if os.environ.get("FEDGATE_FORCE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return bool(sys.stderr.isatty() and sys.stdout.isatty())


def make_progress(iterable: Iterable[Any], *, total: int | None = None, desc: str = "", leave: bool = True) -> Any:
    if not _should_use_tqdm():
        del leave
        return _PlainProgress(iterable, total=total, desc=desc)
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.5,
    )


def progress_write(progress: Any, message: str) -> None:
    writer = getattr(progress, "write", None)
    if callable(writer):
        writer(message)
        if _should_use_tqdm():
            refresh = getattr(progress, "refresh", None)
            if callable(refresh):
                refresh()
    else:
        print(message, flush=True)
