"""overseer.socialjax: a budgeted overseer for SocialJax Harvest (phase 1 of notes/phd_plan/2026-11-15_phased_plan.md).

Each step the overseer checks k of the N agents, looks at what each checked agent did in the last W steps, and
applies a consequence (a reward penalty and/or a freeze) to any agent it finds breaking the rule:
    do not eat an apple that leaves fewer than m apples within two squares
(in Harvest an apple only regrows next to other apples, and never with none nearby).

Needs JAX and SocialJax (scripts/socialjax_local_setup.sh); the rest of the repository does not import this package.
"""
from overseer.socialjax.harvest_overseer import (
    CHOOSERS, HarvestOverseer, OverseerConfig, OverseerState, apples_near, make_harvest_env,
)

__all__ = ["CHOOSERS", "HarvestOverseer", "OverseerConfig", "OverseerState", "apples_near", "make_harvest_env"]
