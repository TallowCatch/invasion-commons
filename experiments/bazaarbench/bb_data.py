"""Load one BazaarBench cell (GPT-5 judge release) into small tables for the overseer replay.

Data: HuggingFace dataset BazaarBench/bazaarbench-analysis-v2-gpt5-judge (CC-BY-4.0),
downloaded by experiments/bazaarbench/download.sh into data/bazaarbench/raw/<cell>/.

Vocabulary used here (the dataset's names in brackets):
- failure types F1-F6 (judge channels T1-T6):
  F1 overstated condition (T1_quality_misrepresentation),
  F2 unowned item (T2_unowned_inventory),
  F3 one item promised to several buyers (T3_inventory_overcommitment),
  F4 premature close (T4_premature_closure),
  F5 personal data or off-platform (T5_externalization_pii),
  F6 fake reputation (T6_unverified_trust_claim);
- stages S1-S5 (considered, attempted, exposed, engaged, realised).

A "bundle" is the unit the GPT-5 judge read: one reasoning summary, one action, one
listing/offer/message text, one thread or one photo, plus context. Each judged
(bundle, channel) pair has a decision; unsafe decisions name the unsafe actor(s) and give
stage evidence with ticks.

A "failure object" is our de-duplication of unsafe decisions: same unsafe actor, same
failure type, and linked through a shared listing, thread, offer, meetup, message, action
or call id (union-find). The README of the release warns that raw decisions are not
de-duplicated incidents; failure objects are our attempt at that.

Time: 1 tick = 2 simulated hours, 12 ticks a day. The continuation window is ticks
361-444, i.e. days 0-6 of the 7-day continuation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CHANNELS = [
    "T1_quality_misrepresentation",
    "T2_unowned_inventory",
    "T3_inventory_overcommitment",
    "T4_premature_closure",
    "T5_externalization_pii",
    "T6_unverified_trust_claim",
]
FTYPE = {c: f"F{i + 1}" for i, c in enumerate(CHANNELS)}
FTYPE_NAME = {
    "F1": "overstated condition",
    "F2": "unowned item",
    "F3": "one item promised twice",
    "F4": "premature close",
    "F5": "personal data / off-platform",
    "F6": "fake reputation",
}
STAGES = ["considered", "attempted", "exposed", "engaged", "realised"]
STAGE_NUM = {s: i + 1 for i, s in enumerate(STAGES)}  # S1..S5
STAGE_NUM["none"] = 0
RECORD_TYPES = {"F1", "F2", "F3"}  # could in principle be checked from inventory / deal records
JUDGE_TYPES = {"F4", "F5", "F6"}
START_TICK = 361
TICKS_PER_DAY = 12
N_DAYS = 7
CHARS_PER_TOKEN = 4.0
LINK_KEYS = ("listing_ids", "thread_ids", "offer_ids", "meetup_ids", "message_ids",
             "action_ids", "call_ids", "photo_ids")
# Evidence that sits only in private reasoning (stage S1 from a reasoning summary).
REASONING_SOURCE_KINDS = {"call_ids"}


def tick_to_day(tick: int) -> int:
    d = (int(tick) - START_TICK) // TICKS_PER_DAY
    return max(0, min(N_DAYS - 1, d))


@dataclass
class Bundle:
    bundle_id: str
    kind: str
    carrier_tick: int
    day: int
    judged_actor_ids: list[int]
    chars: int  # length of the bundle as serialised JSON (what the judge was given, roughly)
    reasoning_chars: int  # characters of private reasoning summaries inside it
    n_actions_by_actor: dict[int, int]
    judge_input_tokens: int = 0  # GPT-5 input tokens allocated to this bundle by the release


@dataclass
class Evidence:
    stage: str
    tick: int
    source_kind: str


@dataclass
class UnsafeDecision:
    bundle_id: str
    channel: str
    actor: int
    max_stage: str
    evidence: list[Evidence]
    links: set[str]
    subtype: str


@dataclass
class FailureObject:
    obj_id: int
    actor: int
    ftype: str
    decisions: list[UnsafeDecision] = field(default_factory=list)

    @property
    def max_stage_num(self) -> int:
        return max(STAGE_NUM.get(d.max_stage, 0) for d in self.decisions)

    def first_day_at_stage(self, stage_num: int) -> int | None:
        """Day the object first reached stage >= stage_num, from evidence ticks."""
        ticks = [e.tick for d in self.decisions for e in d.evidence
                 if STAGE_NUM.get(e.stage, 0) >= stage_num and e.tick is not None]
        if ticks:
            return tick_to_day(min(ticks))
        if self.max_stage_num >= stage_num:  # stage reached but no tick given: use last evidence
            all_ticks = [e.tick for d in self.decisions for e in d.evidence if e.tick is not None]
            return tick_to_day(max(all_ticks)) if all_ticks else None
        return None

    def first_tick_at_stage(self, stage_num: int) -> int | None:
        ticks = [e.tick for d in self.decisions for e in d.evidence
                 if STAGE_NUM.get(e.stage, 0) >= stage_num and e.tick is not None]
        return min(ticks) if ticks else None

    def first_tick_visible(self, view: str = "judge") -> int | None:
        ticks = [e.tick for d in self.decisions for e in d.evidence if e.tick is not None
                 and not (view in ("no_reasoning", "record") and e.source_kind in REASONING_SOURCE_KINDS)]
        if view == "record" and self.ftype not in RECORD_TYPES:
            return None
        return min(ticks) if ticks else None

    def visible_days(self, view: str = "judge") -> set[int]:
        """Days on which a check of this actor would see the failure.

        view="judge": any stage evidence (reasoning summaries included).
        view="no_reasoning": only evidence from actions, messages, listings (not private reasoning).
        view="record": like no_reasoning, and only for F1-F3 (inventory and deal records).
        """
        if view == "record" and self.ftype not in RECORD_TYPES:
            return set()
        days = set()
        for d in self.decisions:
            for e in d.evidence:
                if e.tick is None:
                    continue
                if view in ("no_reasoning", "record") and e.source_kind in REASONING_SOURCE_KINDS:
                    continue
                days.add(tick_to_day(e.tick))
        return days


def _iter_json_lines(path: Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def load_bundles(cell_dir: Path) -> dict[str, Bundle]:
    out: dict[str, Bundle] = {}
    for line in _iter_json_lines(cell_dir / "bundles_bundles.ndjson"):
        b = json.loads(line)
        n_act: dict[int, int] = defaultdict(int)
        for a in b.get("actions", []):
            if a.get("actor_id") is not None and tick_to_day(a.get("tick", b["carrier_tick"])) == tick_to_day(b["carrier_tick"]):
                n_act[int(a["actor_id"])] += 1
        rchars = sum(len(r.get("reasoning_summary") or "") for r in b.get("reasoning", []))
        out[b["bundle_id"]] = Bundle(
            bundle_id=b["bundle_id"],
            kind=b["bundle_kind"],
            carrier_tick=int(b["carrier_tick"]),
            day=tick_to_day(b["carrier_tick"]),
            judged_actor_ids=[int(x) for x in b.get("judged_actor_ids", [])],
            chars=len(line),
            reasoning_chars=rchars,
            n_actions_by_actor=dict(n_act),
        )
    return out


def load_unsafe_decisions(cell_dir: Path, bundles: dict[str, Bundle] | None = None) -> list[UnsafeDecision]:
    """All unsafe (bundle, channel) decisions, one per unsafe actor. Also fills judge tokens."""
    out: list[UnsafeDecision] = []
    for line in _iter_json_lines(cell_dir / "judgments_records.ndjson"):
        r = json.loads(line)
        bid = r["bundle_id"]
        if bundles is not None and bid in bundles:
            bundles[bid].judge_input_tokens = int((r.get("usage") or {}).get("input_tokens") or 0)
        for d in r["decision"]["decisions"]:
            if not d.get("unsafe"):
                continue
            links = {f"{k}:{v}" for k in LINK_KEYS for v in (d.get("source_ids") or {}).get(k, [])}
            ev = [Evidence(stage=e.get("stage"), tick=e.get("tick"), source_kind=e.get("source_kind"))
                  for e in d.get("stage_evidence", [])]
            for actor in d.get("unsafe_actor_ids", []):
                out.append(UnsafeDecision(bundle_id=bid, channel=d["channel"], actor=int(actor),
                                          max_stage=d.get("max_stage", "none"), evidence=ev,
                                          links=links | {f"bundle:{bid}"}, subtype=d.get("subtype", "")))
    return out


def build_failure_objects(decisions: list[UnsafeDecision]) -> list[FailureObject]:
    """Union-find over unsafe decisions sharing (actor, channel) and any linked id."""
    parent = list(range(len(decisions)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_seen: dict[tuple, int] = {}
    for i, d in enumerate(decisions):
        for link in d.links:
            key = (d.actor, d.channel, link)
            if key in first_seen:
                a, b = find(i), find(first_seen[key])
                if a != b:
                    parent[a] = b
            else:
                first_seen[key] = i
    groups: dict[int, list[UnsafeDecision]] = defaultdict(list)
    for i, d in enumerate(decisions):
        groups[find(i)].append(d)
    objs = []
    for n, (_, ds) in enumerate(sorted(groups.items())):
        objs.append(FailureObject(obj_id=n, actor=ds[0].actor, ftype=FTYPE[ds[0].channel], decisions=ds))
    return objs


def load_cell_meta(cell_dir: Path) -> dict:
    led = json.loads((cell_dir / "bundles_ledger.json").read_text())
    return {"cell_id": led["cell_id"], "treated": [int(x) for x in led["cell_binding"]["treated_agent_ids"]],
            "regime": led["cell_binding"]["regime"]}


def load_home_zips(cell_dir: Path, agents: set[int]) -> dict[int, str]:
    """Home zip code per agent from the public-profile snapshots in the bundles."""
    zips: dict[int, str] = {}
    for line in _iter_json_lines(cell_dir / "bundles_bundles.ndjson"):
        if '"home_zip"' not in line:
            continue
        b = json.loads(line)
        for snap in b.get("reputation_snapshots", []):
            prof = snap.get("public_profile") or {}
            aid = prof.get("agent_id")
            if aid is not None and int(aid) in agents and prof.get("home_zip"):
                zips.setdefault(int(aid), str(prof["home_zip"]))
        if len(zips) == len(agents):
            break
    return zips


@dataclass
class Cell:
    cell_id: str
    regime: str
    treated: list[int]
    bundles: dict[str, Bundle]
    decisions: list[UnsafeDecision]
    objects: list[FailureObject]  # treated agents only
    objects_all: list[FailureObject]  # every agent in the market
    zips: dict[int, str]

    def agent_day_cost(self, view: str = "judge") -> dict[tuple[int, int], dict]:
        """Text a check of agent i on day d would read (the bundles where i is judged, that day)."""
        out: dict[tuple[int, int], dict] = defaultdict(lambda: {"bundles": 0, "chars": 0, "reasoning_chars": 0,
                                                                "judge_input_tokens": 0, "actions": 0})
        treated = set(self.treated)
        for b in self.bundles.values():
            for a in b.judged_actor_ids:
                if a not in treated:
                    continue
                row = out[(a, b.day)]
                row["bundles"] += 1
                row["chars"] += b.chars
                row["reasoning_chars"] += b.reasoning_chars
                row["judge_input_tokens"] += b.judge_input_tokens
            for a, n in b.n_actions_by_actor.items():
                if a in treated and b.kind in ("reasoning",):  # each tool call appears once under its reasoning bundle
                    out[(a, b.day)]["actions"] += n
        for a in self.treated:
            for d in range(N_DAYS):
                out[(a, d)]  # make sure every agent-day exists
        return dict(out)


def load_cell(cell_dir: str | Path) -> Cell:
    cell_dir = Path(cell_dir)
    meta = load_cell_meta(cell_dir)
    bundles = load_bundles(cell_dir)
    decisions = load_unsafe_decisions(cell_dir, bundles)
    objs_all = build_failure_objects(decisions)
    treated = set(meta["treated"])
    objs = [o for o in objs_all if o.actor in treated]
    zips = load_home_zips(cell_dir, treated)
    return Cell(cell_id=meta["cell_id"], regime=meta["regime"], treated=meta["treated"], bundles=bundles,
                decisions=decisions, objects=objs, objects_all=objs_all, zips=zips)
