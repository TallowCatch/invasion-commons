"""Seeds for every random draw the overseer makes.

`stable_seed` is copied unchanged from `fishery_sim/calibrated_oversight.py` (the function every runner from R1 to L6
imports), so the overseer draws exactly the same numbers as the stored runs. It is copied, not imported, so that the
overseer never imports a game.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def stable_seed(base: int, *parts) -> int:
    content = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=float)
    return int(base) + int(hashlib.sha256(content.encode()).hexdigest()[:8], 16)


def uniforms(base: int, parts, n: int) -> np.ndarray:
    """n uniform draws in [0, 1) from one generator keyed by (base, *parts)."""
    return np.random.default_rng(stable_seed(base, *parts)).random(n)
