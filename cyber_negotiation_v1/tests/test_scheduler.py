from __future__ import annotations

from collections import Counter

import pytest

from cyberneg.core.scheduler import final_turn_flags, generate_public_schedule


def test_scheduler_no_repeat_and_equal_turns() -> None:
    plan = generate_public_schedule(12, seed=42)
    order = plan.public_order

    assert len(order) == 12
    assert all(a != b for a, b in zip(order, order[1:]))

    counts = Counter(order)
    assert set(counts.values()) == {4}
    assert plan.role_counts == {role: 4 for role in plan.role_counts}


def test_scheduler_rejects_unequal_turn_count_total() -> None:
    with pytest.raises(ValueError):
        generate_public_schedule(5, seed=1)


def test_final_turn_flags_window() -> None:
    assert final_turn_flags(0) == []
    assert final_turn_flags(6, 2) == [False, False, False, False, True, True]

