from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .harvest import HarvestStrategySpec


HARVEST_POLICY_SIGNATURE_FIELDS = [
    "low_patch_threshold",
    "high_patch_threshold",
    "low_harvest_frac",
    "mid_harvest_frac",
    "high_harvest_frac",
    "restraint_low",
    "restraint_high",
    "credit_request_low",
    "credit_request_high",
    "credit_offer_threshold",
    "credit_offer_amount",
    "neighbor_reciprocity_weight",
    "credit_response_weight",
    "cap_compliance_margin",
]

HARVEST_BANK_METADATA_FIELDS = [
    "bank_model_label",
    "bank_provider",
    "bank_model_name",
    "bank_attitude",
    "prompt_nonce",
]

ATTITUDE_CHOICES = ("cooperative", "exploitative")


def sanitize_bank_label(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()


def build_harvest_bank_prompt(attitude: str, patch_max: float, prompt_nonce: int) -> str:
    attitude = attitude.strip().lower()
    if attitude not in ATTITUDE_CHOICES:
        raise ValueError(f"Unknown Harvest bank attitude: {attitude}")
    attitude_instructions = {
        "cooperative": (
            "Generate a cooperative Harvest Commons policy that protects shared patch health, "
            "responds constructively to neighbors, and follows governance caps closely."
        ),
        "exploitative": (
            "Generate an exploitative Harvest Commons policy that prioritizes individual gain, "
            "pushes extraction when possible, and only restrains when pressure is hard to ignore."
        ),
    }
    schema = {
        "rationale": "short explanation, 30 words max",
        "low_patch_threshold": f"float in [0, {patch_max:.1f}]",
        "high_patch_threshold": f"float in [0, {patch_max:.1f}] and >= low_patch_threshold",
        "low_harvest_frac": "float in [0, 1]",
        "mid_harvest_frac": "float in [0, 1]",
        "high_harvest_frac": "float in [0, 1]",
        "restraint_low": "float in [0, 1]",
        "restraint_high": "float in [0, 1]",
        "credit_request_low": "float in [0, 1]",
        "credit_request_high": "float in [0, 1]",
        "credit_offer_threshold": f"float in [0, {patch_max:.1f}]",
        "credit_offer_amount": "float in [0, 1]",
        "neighbor_reciprocity_weight": "float in [0, 1]",
        "credit_response_weight": "float in [0, 1]",
        "cap_compliance_margin": "float in [0, 0.25]",
    }
    return (
        "You are writing a complete Harvest Commons strategy as strict JSON.\n"
        f"Attitude: {attitude}\n"
        f"Variation nonce: {prompt_nonce}\n"
        f"{attitude_instructions[attitude]}\n"
        "The policy should be internally consistent and executable as a threshold policy.\n"
        "Output exactly one JSON object with no markdown and no extra keys.\n"
        f"Schema:\n{schema}\n"
    )


def harvest_policy_signature(spec_or_row: HarvestStrategySpec | pd.Series | dict[str, Any]) -> tuple[float, ...]:
    if isinstance(spec_or_row, HarvestStrategySpec):
        values = [getattr(spec_or_row, field) for field in HARVEST_POLICY_SIGNATURE_FIELDS]
    elif isinstance(spec_or_row, pd.Series):
        values = [spec_or_row[field] for field in HARVEST_POLICY_SIGNATURE_FIELDS]
    else:
        values = [spec_or_row[field] for field in HARVEST_POLICY_SIGNATURE_FIELDS]
    return tuple(round(float(value), 6) for value in values)


def strategy_spec_to_bank_row(
    spec: HarvestStrategySpec,
    *,
    bank_model_label: str,
    bank_provider: str,
    bank_model_name: str,
    bank_attitude: str,
    prompt_nonce: int,
) -> dict[str, Any]:
    row = asdict(spec)
    row.update(
        {
            "bank_model_label": bank_model_label,
            "bank_provider": bank_provider,
            "bank_model_name": bank_model_name,
            "bank_attitude": bank_attitude,
            "prompt_nonce": int(prompt_nonce),
            "policy_signature": "|".join(f"{value:.6f}" for value in harvest_policy_signature(spec)),
        }
    )
    return row


def bank_row_to_strategy_spec(row: pd.Series | dict[str, Any], strategy_id: str | None = None) -> HarvestStrategySpec:
    if isinstance(row, pd.Series):
        def get_value(key: str, default: Any = None) -> Any:
            return row[key] if key in row.index else default
    else:
        def get_value(key: str, default: Any = None) -> Any:
            return row.get(key, default)
    return HarvestStrategySpec(
        strategy_id=strategy_id or str(get_value("strategy_id")),
        low_patch_threshold=float(get_value("low_patch_threshold")),
        high_patch_threshold=float(get_value("high_patch_threshold")),
        low_harvest_frac=float(get_value("low_harvest_frac")),
        mid_harvest_frac=float(get_value("mid_harvest_frac")),
        high_harvest_frac=float(get_value("high_harvest_frac")),
        restraint_low=float(get_value("restraint_low")),
        restraint_high=float(get_value("restraint_high")),
        credit_request_low=float(get_value("credit_request_low")),
        credit_request_high=float(get_value("credit_request_high")),
        credit_offer_threshold=float(get_value("credit_offer_threshold")),
        credit_offer_amount=float(get_value("credit_offer_amount")),
        neighbor_reciprocity_weight=float(get_value("neighbor_reciprocity_weight")),
        credit_response_weight=float(get_value("credit_response_weight")),
        cap_compliance_margin=float(get_value("cap_compliance_margin")),
        origin=str(get_value("origin", "llm_bank")),
        rationale=str(get_value("rationale", "")),
        llm_parse_status=str(get_value("llm_parse_status", "")),
        llm_parse_error_type=str(get_value("llm_parse_error_type", "")),
    )


def load_harvest_strategy_bank(
    bank_csv: str,
    *,
    model_label: str | None = None,
    attitude: str | None = None,
) -> pd.DataFrame:
    bank_df = pd.read_csv(bank_csv)
    if model_label:
        bank_df = bank_df[bank_df["bank_model_label"] == model_label].copy()
    if attitude:
        bank_df = bank_df[bank_df["bank_attitude"] == attitude].copy()
    if "policy_signature" not in bank_df.columns:
        bank_df["policy_signature"] = bank_df.apply(harvest_policy_signature, axis=1).map(
            lambda values: "|".join(f"{value:.6f}" for value in values)
        )
    return bank_df.reset_index(drop=True)


def sample_population_from_bank(
    bank_df: pd.DataFrame,
    *,
    population_size: int,
    exploitative_share: float,
    rng: np.random.Generator,
    model_label: str,
) -> tuple[list[HarvestStrategySpec], list[str]]:
    if bank_df.empty:
        raise ValueError("Strategy bank is empty.")
    coop_df = bank_df[(bank_df["bank_model_label"] == model_label) & (bank_df["bank_attitude"] == "cooperative")].copy()
    exp_df = bank_df[(bank_df["bank_model_label"] == model_label) & (bank_df["bank_attitude"] == "exploitative")].copy()
    if coop_df.empty or exp_df.empty:
        raise ValueError(f"Missing cooperative/exploitative bank entries for model_label={model_label}")

    n_exploitative = int(round(population_size * float(np.clip(exploitative_share, 0.0, 1.0))))
    n_cooperative = population_size - n_exploitative
    sampled_rows: list[pd.Series] = []
    sampled_attitudes: list[str] = []

    for attitude, count, source_df in [
        ("cooperative", n_cooperative, coop_df),
        ("exploitative", n_exploitative, exp_df),
    ]:
        if count <= 0:
            continue
        replace = len(source_df) < count
        sample = source_df.sample(n=count, replace=replace, random_state=int(rng.integers(0, 1_000_000_000)))
        for _, row in sample.iterrows():
            sampled_rows.append(row)
            sampled_attitudes.append(attitude)

    order = rng.permutation(len(sampled_rows))
    population: list[HarvestStrategySpec] = []
    ordered_attitudes: list[str] = []
    for i, idx in enumerate(order):
        row = sampled_rows[int(idx)]
        ordered_attitudes.append(sampled_attitudes[int(idx)])
        population.append(bank_row_to_strategy_spec(row, strategy_id=f"bank_s{i}"))
    return population, ordered_attitudes


def sample_strategy_from_gene_bank(
    bank_df: pd.DataFrame,
    *,
    model_label: str,
    attitude: str,
    rng: np.random.Generator,
    strategy_id: str,
) -> HarvestStrategySpec:
    subset = bank_df[(bank_df["bank_model_label"] == model_label) & (bank_df["bank_attitude"] == attitude)].copy()
    if subset.empty:
        raise ValueError(f"No bank rows for model_label={model_label}, attitude={attitude}")
    row = subset.sample(n=1, random_state=int(rng.integers(0, 1_000_000_000))).iloc[0]
    return bank_row_to_strategy_spec(row, strategy_id=strategy_id)
