from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
PHASE0_DATA = DATA_ROOT / "phase0_yahoo"
PHASE0_AUDIT = DATA_ROOT / "phase0_audit"
PHASE1_DATA = DATA_ROOT / "phase1"
PHASE1_AUDIT = DATA_ROOT / "phase1_audit"
TRIALS_DIR = PROJECT_ROOT / "trials"


def ensure_dirs() -> None:
    for p in (DATA_ROOT, PHASE0_DATA, PHASE0_AUDIT, PHASE1_DATA, PHASE1_AUDIT, TRIALS_DIR):
        p.mkdir(parents=True, exist_ok=True)
