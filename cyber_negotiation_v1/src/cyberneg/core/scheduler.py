from __future__ import annotations

import random
from collections import Counter

from .enums import ROLE_ORDER, RoleId
from .schemas import SchedulerPlan


def round0_roles() -> list[RoleId]:
    return list(ROLE_ORDER)


def _feasible(counts: dict[RoleId, int]) -> bool:
    remaining = sum(counts.values())
    if remaining <= 1:
        return True
    max_count = max(counts.values(), default=0)
    return max_count <= (remaining - max_count) + 1


def _backtrack_build(
    rng: random.Random,
    roles: list[RoleId],
    counts: dict[RoleId, int],
    last_role: RoleId | None,
    seq: list[RoleId],
    total: int,
) -> bool:
    if len(seq) == total:
        return True
    candidates = [r for r in roles if counts.get(r, 0) > 0 and r != last_role]
    rng.shuffle(candidates)
    for role in candidates:
        counts[role] -= 1
        if _feasible(counts):
            seq.append(role)
            if _backtrack_build(rng, roles, counts, role, seq, total):
                return True
            seq.pop()
        counts[role] += 1
    return False


def final_turn_flags(total_public_messages: int, final_window: int = 1) -> list[bool]:
    if total_public_messages <= 0:
        return []
    final_window = max(1, min(final_window, total_public_messages))
    start = total_public_messages - final_window
    return [idx >= start for idx in range(total_public_messages)]


def generate_public_schedule(total_public_messages: int, seed: int) -> SchedulerPlan:
    if total_public_messages <= 0:
        raise ValueError("total_public_messages must be > 0")
    if total_public_messages % len(ROLE_ORDER) != 0:
        raise ValueError("total_public_messages must be divisible by 3 to ensure equal turns")

    quota = total_public_messages // len(ROLE_ORDER)
    counts: dict[RoleId, int] = {role: quota for role in ROLE_ORDER}
    roles = list(ROLE_ORDER)
    seq: list[RoleId] = []
    rng = random.Random(seed)

    if not _backtrack_build(rng, roles, counts, None, seq, total_public_messages):
        raise RuntimeError("Could not build a valid public schedule with fairness/no-repeat constraints")

    role_counts = Counter(seq)
    return SchedulerPlan(
        order_seed=seed,
        public_order=seq,
        final_turn_flags=final_turn_flags(total_public_messages, 1),
        role_counts={role: int(role_counts.get(role, 0)) for role in ROLE_ORDER},
    )

