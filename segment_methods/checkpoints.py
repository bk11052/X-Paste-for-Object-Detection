from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_checkpoint(candidates: list[str | Path], label: str) -> str:
    """Return the first existing checkpoint path from candidates."""
    root = repo_root()
    tried: list[str] = []
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if not p.is_absolute():
            p = root / p
        tried.append(str(p))
        if p.exists():
            return str(p)
    msg = [f"Missing checkpoint for {label}. Tried:"]
    msg.extend(f"  - {p}" for p in tried)
    raise FileNotFoundError("\n".join(msg))
