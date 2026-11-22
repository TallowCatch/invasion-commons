"""BazaarBench judging set: evidence bundles with the GPT-5 judge's verdict for one failure type.

Data: the offline-replay download (experiments/bazaarbench/download.sh -> data/bazaarbench/raw/<cell>/), the
HuggingFace release BazaarBench/bazaarbench-analysis-v2-gpt5-judge (CC-BY-4.0). Schema notes:
notes/bazaarbench/offline_replay/protocol.md and experiments/bazaarbench/bb_data.py.

One example is one (bundle, failure type) pair the GPT-5 judge decided: "unsafe" (label 1) or not (label 0). A bundle
is what GPT-5 read in one go: one reasoning summary, one action, one listing/offer/message text, one thread or one
photo, with context. GPT-5 counted a failure that was only planned in private reasoning (stage S1, "considered") as
unsafe, so `reasoning_only` marks unsafe examples whose evidence is all private reasoning.

Two renderings:
- bundle: the bundle as compact JSON with bookkeeping fields (hashes, routing, pools) removed: what GPT-5 saw;
- public: the same without private reasoning summaries (what a check of public actions and messages would read).
  On reasoning-only failures the public rendering cannot show the failure, so misses there are expected (claim C4).

Sampling: up to `per_cell` unsafe and the same number of safe decisions for each failure type, from both cells
(ordinary L1 and exploit L3 instructions, tested model gpt-oss-120b), only bundles whose rendering fits in
`max_chars`. F6 has very few unsafe decisions (2-4 per cell), so it is small.

The GPT-5 rubric itself is not in the release; the rule texts in prompts.py are our paraphrase of the six failure
types. So alpha and beta here measure agreement with GPT-5, not truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHANNELS = ("T1_quality_misrepresentation", "T2_unowned_inventory", "T3_inventory_overcommitment",
            "T4_premature_closure", "T5_externalization_pii", "T6_unverified_trust_claim")
FTYPE = {c: f"F{i + 1}" for i, c in enumerate(CHANNELS)}
CELLS = ("gpt55_L1_gptoss120b", "gpt55_L3_gptoss120b")
REASONING_SOURCE_KINDS = {"call_ids"}  # as in experiments/bazaarbench/bb_data.py
DROP_KEYS = {
    "digest", "content_hash", "eligible_pool_ids", "episode_routes", "denominator_routes", "denominator_kinds",
    "prefilter_candidate", "prefilter_candidate_by_channel", "schema_version", "cell_id", "link_confidence",
    "event_id", "terminal_mutable_context", "metadata", "reasoning_authoritative", "reasoning_coverage",
    "target_channels", "bundle_id", "carrier_id", "tool_index", "observed",
}
KEEP_ORDER = ("bundle_kind", "carrier_kind", "carrier_tick", "judged_actor_ids", "treated_actor_ids",
              "counterparty_ids", "actions", "reasoning", "observable", "encounters", "reputation_snapshots")


def prune(obj):
    """Drop bookkeeping keys and empty values, recursively."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in DROP_KEYS:
                continue
            v = prune(v)
            if v in (None, "", [], {}):
                continue
            out[k] = v
        return out
    if isinstance(obj, list):
        return [x for x in (prune(v) for v in obj) if x not in (None, "", [], {})]
    return obj


def render_bundle(bundle: dict, public: bool = False) -> str:
    b = dict(bundle)
    if public:
        b.pop("reasoning", None)
    keys = [k for k in KEEP_ORDER if k in b] + sorted(k for k in b if k not in KEEP_ORDER)
    pruned = prune({k: b[k] for k in keys})
    return json.dumps(pruned, ensure_ascii=False, separators=(",", ":"))


def load_decisions(cell_dir: Path) -> list[dict]:
    """Every (bundle, channel) decision in a cell, with the GPT-5 label."""
    out = []
    with open(cell_dir / "judgments_records.ndjson") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            for d in r["decision"]["decisions"]:
                kinds = {e.get("source_kind") for e in d.get("stage_evidence", [])}
                out.append(dict(bundle_id=r["bundle_id"], channel=d["channel"], label=int(bool(d.get("unsafe"))),
                                gpt5_max_stage=d.get("max_stage", "none"),
                                gpt5_unsafe_actors=[int(x) for x in d.get("unsafe_actor_ids", [])],
                                gpt5_subtype=d.get("subtype", ""),
                                reasoning_only=bool(d.get("unsafe")) and bool(kinds) and kinds <= REASONING_SOURCE_KINDS))
    return out


def load_bundles(cell_dir: Path, wanted: set[str] | None = None) -> dict[str, dict]:
    out = {}
    with open(cell_dir / "bundles_bundles.ndjson") as fh:
        for line in fh:
            if not line.strip():
                continue
            b = json.loads(line)
            if wanted is None or b["bundle_id"] in wanted:
                out[b["bundle_id"]] = b
    return out


def build_dataset(raw_dir="data/bazaarbench/raw", cells=CELLS, per_cell=50, max_chars=16000, seed=0):
    """Balanced examples: for each failure type and cell, up to `per_cell` unsafe and as many safe decisions."""
    rng = np.random.default_rng(seed)
    examples, manifest = [], dict(seed=seed, per_cell=per_cell, max_chars=max_chars, cells={})
    for cell in cells:
        cdir = Path(raw_dir) / cell
        decisions = load_decisions(cdir)
        bundles = load_bundles(cdir)
        rendered = {bid: render_bundle(b) for bid, b in bundles.items()}
        manifest["cells"][cell] = {}
        for ch in CHANNELS:
            fits = [d for d in decisions if d["channel"] == ch and d["bundle_id"] in rendered
                    and len(rendered[d["bundle_id"]]) <= max_chars]
            unsafe = [d for d in fits if d["label"] == 1]
            safe = [d for d in fits if d["label"] == 0]
            n = min(per_cell, len(unsafe))
            pick = [unsafe[i] for i in rng.choice(len(unsafe), n, replace=False)] if n else []
            pick += [safe[i] for i in rng.choice(len(safe), n, replace=False)] if n else []
            manifest["cells"][cell][FTYPE[ch]] = dict(unsafe_available=len(unsafe), safe_available=len(safe),
                                                      chosen_each=n)
            for d in pick:
                b = bundles[d["bundle_id"]]
                examples.append(dict(
                    dataset="bazaar", cell=cell, ftype=FTYPE[ch], **d,
                    judged_actor_ids=[int(x) for x in b.get("judged_actor_ids", [])],
                    bundle_kind=b.get("bundle_kind"),
                    renderings=dict(bundle=rendered[d["bundle_id"]], public=render_bundle(b, public=True))))
    for i, ex in enumerate(examples):
        ex["id"] = f"bazaar_{ex['ftype']}_{i:05d}"
    return examples, manifest
