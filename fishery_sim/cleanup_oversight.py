"""Optional, pinned SocialJax Cleanup sidecar; no replacement game dynamics.

Importing this module needs only the standard library. See CLEANUP_ADAPTER.md
for installation, the productive-state target, and limits on evidence/authority.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib
import importlib.metadata
import json
import math
from numbers import Integral
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence


SOCIALJAX_URL = "https://github.com/cooperativex/SocialJax.git"
SOCIALJAX_REVISION = "9df972d07657c7e2d4cd4fec1ed1f437099cb128"
NOOP = 6  # Native Actions.stay, checked against the imported enum.
CLEAN = 8
NUM_ACTIONS = 9
NUM_AGENTS = 7
SNAPSHOT_VERSION = 1


class CleanupUnavailable(RuntimeError):
    """The optional dependencies or verified upstream checkout are unavailable."""


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def validate_actions(actions: Sequence[int]) -> tuple[int, ...]:
    """Require one native discrete action per agent, in native agent order."""
    if len(actions) != NUM_AGENTS:
        raise ValueError(f"expected {NUM_AGENTS} actions")
    return tuple(_integer(a, "action", 0, NUM_ACTIONS - 1) for a in actions)


def replace_with_noop(
    actions: Sequence[int], rejected_agents: Iterable[int]
) -> tuple[int, ...]:
    """The entire intervention menu: keep a proposal or replace it with stay.

    No state edits, relocation, forced cleaning, or reward changes. A noop does
    not pause exogenous pollution or undo other agents' actions.
    """
    executed = list(validate_actions(actions))
    for agent in rejected_agents:
        executed[_integer(agent, "agent", 0, NUM_AGENTS - 1)] = NOOP
    return tuple(executed)


def effective_spawn_probability(
    dirt_count: int,
    river_cells: int,
    *,
    maximum_rate: float = 0.05,
    depletion: float = 0.4,
    restoration: float = 0.0,
) -> float:
    """Probability implied by upstream's uniform < growth-threshold test.

    Upstream only clips interpolation above at 1; a negative threshold cannot
    spawn an apple, so the effective probability is clipped below at zero here.
    This is a diagnostic formula, not a substitute for native transitions.
    """
    river_cells = _integer(river_cells, "river_cells", 1, 2**31 - 1)
    dirt_count = _integer(dirt_count, "dirt_count", 0, river_cells)
    if not (0 < maximum_rate <= 1 and 0 <= restoration < depletion <= 1):
        raise ValueError("invalid native growth parameters")
    interpolation = (dirt_count / river_cells - depletion) / (restoration - depletion)
    return maximum_rate * max(0.0, min(interpolation, 1.0))


@dataclass(frozen=True)
class CleanupConfig:
    horizon: int = 300
    minimum_spawn_fraction: float = 0.5

    def __post_init__(self) -> None:
        _integer(self.horizon, "horizon", 1, 1_000_000)
        if isinstance(self.minimum_spawn_fraction, bool) or not (
            0 < self.minimum_spawn_fraction <= 1
        ):
            raise ValueError("minimum_spawn_fraction must be in (0, 1]")


@dataclass(frozen=True, eq=False)
class CleanupState:
    native: Any
    observations: Any
    contract: str


@dataclass(frozen=True, eq=False)
class LocalEvidence:
    """Only one native observation and that agent's proposal, no state summary."""

    agent_id: int
    observation: Any
    proposed_action: int


@dataclass(frozen=True, eq=False)
class FullStateEvidence:
    """Privileged evaluator/reference evidence, NOT pooled local observations."""

    state: CleanupState
    proposed_actions: tuple[int, ...]


def _verified_source(source_path: str | Path | None) -> Path:
    source = source_path or os.environ.get("SOCIALJAX_SOURCE")
    guidance = (
        "Install requirements-cleanup.txt in an isolated env and set "
        f"SOCIALJAX_SOURCE to a clean checkout of {SOCIALJAX_URL} "
        f"at {SOCIALJAX_REVISION}. See notes/research_review/CLEANUP_ADAPTER.md."
    )
    if not source:
        raise CleanupUnavailable(guidance)
    root = Path(source).expanduser().resolve()
    try:
        def git(*args: str) -> str:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.PIPE,
                text=True, timeout=10,
            ).strip()

        if git("rev-parse", "HEAD") != SOCIALJAX_REVISION:
            raise CleanupUnavailable(f"SocialJax revision mismatch. {guidance}")
        if git("status", "--porcelain", "--untracked-files=all", "--", "socialjax"):
            raise CleanupUnavailable(f"SocialJax source is modified. {guidance}")
        if not (root / "socialjax/environments/cleanup/clean_up.py").is_file():
            raise CleanupUnavailable(guidance)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CleanupUnavailable(guidance) from exc
    return root


class CleanupAdapter:
    """Host-side interface around the unmodified, seven-agent native game.

    The sidecar owns the episode cutoff. Native num_inner_steps=horizon+1
    avoids upstream's automatic reset and zeroed reward on our last step.
    All physical transitions through horizon are native step_env calls.
    """

    def __init__(
        self, config: CleanupConfig | None = None, *, source_path: str | Path | None = None
    ) -> None:
        self.config = config or CleanupConfig()
        self.source_path = _verified_source(source_path)
        sys.path.insert(0, str(self.source_path))
        try:
            self.jax = importlib.import_module("jax")
            self.np = importlib.import_module("numpy")
            native = importlib.import_module("socialjax.environments.cleanup.clean_up")
        except ImportError as exc:
            raise CleanupUnavailable(
                "Cleanup optional dependency import failed; install requirements-cleanup.txt "
                f"in the isolated environment. Original error: {exc}"
            ) from exc
        finally:
            sys.path.remove(str(self.source_path))
        expected = self.source_path / "socialjax/environments/cleanup/clean_up.py"
        if Path(native.__file__).resolve() != expected:
            raise CleanupUnavailable("A different SocialJax module is already imported")
        if (int(native.Actions.stay), int(native.Actions.zap_clean), len(native.Actions)) != (
            NOOP, CLEAN, NUM_ACTIONS
        ):
            raise CleanupUnavailable("Unexpected native action enum")
        self._native_type = native.State
        self._items = native.Items
        self.env = native.Clean_up(
            num_agents=NUM_AGENTS, num_inner_steps=self.config.horizon + 1,
            num_outer_steps=1, shared_rewards=False, jit=True,
        )
        self.manifest = {
            "snapshot_version": SNAPSHOT_VERSION,
            "source_url": SOCIALJAX_URL,
            "source_revision": SOCIALJAX_REVISION,
            "config": asdict(self.config),
            "native_kwargs": {
                "num_agents": NUM_AGENTS, "num_inner_steps": self.config.horizon + 1,
                "num_outer_steps": 1, "shared_rewards": False, "jit": True,
            },
            "runtime": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "flax", "chex", "numpy")
            },
            "backend": self.jax.default_backend(),
            "jax_enable_x64": bool(self.jax.config.jax_enable_x64),
            "prng_impl": str(self.jax.config.jax_default_prng_impl),
        }
        self._contract = hashlib.sha256(self._json(self.manifest)).hexdigest()
        self._template: CleanupState | None = None

    @staticmethod
    def _json(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def _key(self, seed: int) -> Any:
        return self.jax.random.PRNGKey(_integer(seed, "seed", 0, 2**32 - 1))

    def _check_state(self, state: CleanupState) -> None:
        if not isinstance(state, CleanupState) or state.contract != self._contract:
            raise ValueError("state belongs to a different Cleanup contract")
        if not isinstance(state.native, self._native_type):
            raise ValueError("expected native Cleanup State")
        if not (0 <= int(state.native.inner_t) <= self.config.horizon) or int(state.native.outer_t):
            raise ValueError("state time is outside the sidecar episode")

    def reset(self, seed: int) -> CleanupState:
        observations, native = self.env.reset(self._key(seed))
        state = CleanupState(native, observations, self._contract)
        self._template = state
        return state

    def transition(
        self, state: CleanupState, actions: Sequence[int], seed: int
    ) -> tuple[CleanupState, tuple[float, ...], bool]:
        self._check_state(state)
        if int(state.native.inner_t) >= self.config.horizon:
            raise ValueError("terminal state: call reset explicitly")
        observations, native, rewards, done, _ = self.env.step_env(
            self._key(seed), state.native, self.jax.numpy.asarray(validate_actions(actions)),
        )
        if bool(done["__all__"]):
            raise RuntimeError("unexpected native auto-reset inside sidecar horizon")
        result = CleanupState(native, observations, self._contract)
        # Keep upstream's individual reward scale: seven units per apple.
        return result, tuple(float(r) for r in rewards), int(native.inner_t) == self.config.horizon

    def metrics(self, state: CleanupState) -> dict[str, int | float]:
        """Evaluator-only metrics; not permitted local monitor evidence."""
        self._check_state(state)
        native = state.native
        dirt = int(self.np.count_nonzero(self.np.asarray(native.potential_dirt_and_dirt_label) == self._items.dirt))
        river = len(native.potential_dirt_and_dirt_locs) + len(self.env.RIVER)
        probability = effective_spawn_probability(
            dirt, river, maximum_rate=self.env.maxAppleGrowthRate,
            depletion=self.env.thresholdDepletion, restoration=self.env.thresholdRestoration,
        )
        return {
            "step": int(native.inner_t), "dirt_count": dirt, "river_cells": river,
            "dirt_fraction": dirt / river, "effective_spawn_probability": probability,
            "minimum_spawn_probability": self.config.minimum_spawn_fraction * self.env.maxAppleGrowthRate,
            "apples_on_grid": int(self.np.count_nonzero(self.np.asarray(native.grid) == self._items.apple)),
        }

    def safe(self, state: CleanupState) -> bool:
        """At least the declared fraction of maximum native apple-spawn rate.

        A productive-capacity state predicate, not a viability certificate or
        evidence of actual harvesting. Default threshold corresponds to f<=0.2.
        """
        metrics = self.metrics(state)
        return metrics["effective_spawn_probability"] >= metrics["minimum_spawn_probability"]

    def local_evidence(
        self, state: CleanupState, actions: Sequence[int], agent_id: int
    ) -> LocalEvidence:
        self._check_state(state)
        agent_id = _integer(agent_id, "agent_id", 0, NUM_AGENTS - 1)
        proposed = validate_actions(actions)
        return LocalEvidence(agent_id, state.observations[agent_id], proposed[agent_id])

    def full_state_evidence(
        self, state: CleanupState, actions: Sequence[int]
    ) -> FullStateEvidence:
        self._check_state(state)
        return FullStateEvidence(state, validate_actions(actions))

    def snapshot(self, state: CleanupState) -> bytes:
        """Canonical JSON, complete native state + observations; no pickle/RNG.

        Replay requires the same contract and explicitly supplied future seeds.
        Controller memory is external; the smoke controller is stateless.
        """
        self._check_state(state)

        def encode(value: Any) -> dict[str, Any]:
            array = self.np.asarray(value)
            return {"dtype": array.dtype.str, "shape": list(array.shape),
                    "data": base64.b64encode(array.tobytes(order="C")).decode("ascii")}

        return self._json({
            "manifest": self.manifest,
            "native": {f.name: encode(getattr(state.native, f.name)) for f in fields(state.native)},
            "observations": encode(state.observations),
        })

    def restore(self, snapshot: bytes) -> CleanupState:
        """Restore same-runtime snapshots with strict field/shape/dtype checks."""
        if len(snapshot) > 2_000_000:
            raise ValueError("Cleanup snapshot is too large")
        payload = json.loads(snapshot)
        if set(payload) != {"manifest", "native", "observations"} or payload["manifest"] != self.manifest:
            raise ValueError("snapshot contract mismatch")
        template = self._template if self._template is not None else self.reset(0)
        names = {f.name for f in fields(template.native)}
        if set(payload["native"]) != names:
            raise ValueError("snapshot native fields mismatch")

        def decode(record: dict[str, Any], reference: Any) -> Any:
            expected = self.np.asarray(reference)
            if set(record) != {"dtype", "shape", "data"} or (
                record["dtype"] != expected.dtype.str or record["shape"] != list(expected.shape)
            ):
                raise ValueError("snapshot array schema mismatch")
            raw = base64.b64decode(record["data"], validate=True)
            if len(raw) != math.prod(expected.shape) * expected.dtype.itemsize:
                raise ValueError("snapshot array length mismatch")
            array = self.np.frombuffer(raw, dtype=expected.dtype).reshape(expected.shape)
            return self.jax.numpy.asarray(array)

        state = CleanupState(
            self._native_type(**{name: decode(payload["native"][name], getattr(template.native, name)) for name in names}),
            decode(payload["observations"], template.observations), self._contract,
        )
        self._check_state(state)
        return state
