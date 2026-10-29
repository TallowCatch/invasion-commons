from __future__ import annotations


CONDITION_LABELS = {
    "none": "No oversight",
    "bottom_up_only": "Local oversight",
    "top_down_only": "Global signal",
    "hybrid": "Hybrid oversight",
}

SCENARIO_LABELS = {
    "community_irrigation": "Moderate-coupling commons",
    "forest_co_management": "High-coupling commons",
    "regulated_fishery": "Regulated commons",
}

ACTOR_CAPABILITY_LABELS = {
    "low_actor": "Low actor capability",
    "medium_actor": "Medium actor capability",
    "high_actor": "High actor capability",
}

OVERSEER_CAPABILITY_LABELS = {
    "strong_overseer": "Strong overseer",
    "limited_overseer": "Limited overseer",
    "weak_overseer": "Weak overseer",
}


def condition_label(value: str) -> str:
    text = str(value)
    return CONDITION_LABELS.get(text, text.replace("_", " ").title())


def scenario_label(value: str) -> str:
    text = str(value)
    return SCENARIO_LABELS.get(text, text.replace("_", " ").title())


def actor_capability_label(value: str) -> str:
    text = str(value)
    return ACTOR_CAPABILITY_LABELS.get(text, text.replace("_", " ").title())


def overseer_capability_label(value: str) -> str:
    text = str(value)
    return OVERSEER_CAPABILITY_LABELS.get(text, text.replace("_", " ").title())
