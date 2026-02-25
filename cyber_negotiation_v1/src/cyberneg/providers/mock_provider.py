from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any

from ..core.enums import RoleId, Severity
from .base import BaseProvider, ProviderCallContext, ProviderResponse


class MockProvider(BaseProvider):
    """Deterministic offline provider for V1 scaffold tests and dry-runs."""

    def __init__(
        self,
        provider_name: str = "mock",
        model_name: str = "mock-deterministic-v1",
        *,
        invalid_json_attempts_per_turn: int = 0,
        deterministic_seed_offset: int = 0,
    ) -> None:
        super().__init__(provider_name=provider_name, provider_kind="mock", model_name=model_name)
        self.invalid_json_attempts_per_turn = max(0, int(invalid_json_attempts_per_turn))
        self.deterministic_seed_offset = int(deterministic_seed_offset)
        self._attempts_by_turn: dict[str, int] = defaultdict(int)

    def generate(self, prompt: str, ctx: ProviderCallContext) -> ProviderResponse:
        self._attempts_by_turn[ctx.turn_id] += 1
        attempt = self._attempts_by_turn[ctx.turn_id]
        if attempt <= self.invalid_json_attempts_per_turn:
            bad = '{"private_notes": "oops", "assessment": '  # intentionally invalid JSON
            return ProviderResponse(
                text=bad,
                usage={"input_tokens": 100, "output_tokens": 5, "total_tokens": 105},
                request_metadata={"mock_attempt": attempt},
                response_metadata={"mock_invalid": True},
            )

        payload = self._build_valid_payload(prompt, ctx, attempt)
        return ProviderResponse(
            text=json.dumps(payload, ensure_ascii=False),
            usage={"input_tokens": 150, "output_tokens": 220, "total_tokens": 370},
            request_metadata={"mock_attempt": attempt},
            response_metadata={"mock_invalid": False},
        )

    def _build_valid_payload(self, prompt: str, ctx: ProviderCallContext, attempt: int) -> dict[str, Any]:
        md = ctx.metadata or {}
        line_ids = list(md.get("line_ids") or ["L001"])
        role = ctx.role_id.value if ctx.role_id else "R"
        public_turn_index = ctx.public_turn_index if ctx.public_turn_index is not None else -1
        final_turn = bool(md.get("is_final_public_turn", False))
        label_set = list(md.get("label_set_labels") or ["Other", "NoFinding"])

        # Conservative deterministic mock heuristic
        preferred = "XSS_Reflected" if "XSS_Reflected" in label_set else label_set[0]
        alt_1 = "NoFinding" if "NoFinding" in label_set else (label_set[1] if len(label_set) > 1 else preferred)
        alt_2 = "XSS_Stored" if "XSS_Stored" in label_set else (label_set[2] if len(label_set) > 2 else preferred)

        if ctx.phase.value == "round0":
            if role == RoleId.C.value:
                sev = Severity.LOW
            else:
                sev = Severity.MEDIUM
        else:
            # Negotiate toward exact consensus by the final turn.
            sev = Severity.MEDIUM if final_turn or public_turn_index >= 2 else (Severity.LOW if role == "C" else Severity.MEDIUM)

        ranked = [
            {
                "rank": 1,
                "label": preferred,
                "severity": sev.value,
                "confidence": 0.76,
                "citations": line_ids[:2] if len(line_ids) >= 2 else line_ids[:1],
                "rationale": "Primary hypothesis is supported by reflected input behavior and PoC rendering evidence.",
            },
            {
                "rank": 2,
                "label": alt_2,
                "severity": "Low",
                "confidence": 0.31,
                "citations": line_ids[2:3],
                "rationale": "Alternative XSS variant remains possible but evidence is weaker than the reflected case.",
            },
            {
                "rank": 3,
                "label": alt_1,
                "severity": "Info",
                "confidence": 0.18,
                "citations": [],
                "rationale": "A conservative no-finding interpretation is possible but does not explain the PoC evidence well.",
            },
        ]

        seed_material = f"{ctx.turn_id}|{self.deterministic_seed_offset}|{attempt}"
        stable_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(stable_seed)
        if ctx.phase.value == "baseline":
            msg = (
                "My current assessment prioritizes a reflected cross-site scripting finding because the packet shows "
                "unsafely reflected user-controlled input and a browser-rendered proof of execution. I am keeping "
                "the severity at Medium pending stronger impact confirmation, while still citing the exact packet lines "
                "that most directly support exploitability and report defensibility."
            )
        elif ctx.phase.value == "round0":
            msg = (
                f"As {role}, my independent starting position is to keep the top hypothesis evidence-led and explicitly "
                "tied to the packet lines showing reflected input handling and browser execution evidence. I currently "
                "treat the severity as conservative-to-moderate and expect negotiation to refine confidence and report "
                "defensibility rather than replace the core finding type without new evidence."
            )
        else:
            action = "commit" if final_turn else "propose"
            phrase = rng.choice(["tighten", "stabilize", "clarify", "defend"])
            msg = (
                f"I {action} to a reflected XSS-centered conclusion and want to {phrase} the rationale around the packet "
                "evidence instead of widening scope prematurely. The strongest support remains the reflected parameter "
                "behavior plus browser execution proof, so I prefer converging on a defensible Medium severity unless "
                "someone can show a stronger impact argument directly tied to the cited lines."
            )

        return {
            "private_notes": f"Private notes for {role}: track competing hypotheses, severity disagreement, and citation quality.",
            "private_plan": (
                "Next move: keep top-1 stable if evidence remains strongest; ask for precise citation-backed objections; "
                + ("final turn so clearly commit to one defensible label/severity." if final_turn else "prepare a concise convergence proposal.")
            ),
            "public_message": msg,
            "assessment": {
                "ranked_findings": ranked,
                "decision_summary": "Reflected XSS remains the most defensible label from current evidence; severity calibrated conservatively.",
            },
        }
