from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cyberneg.providers.anthropic import AnthropicProvider
from cyberneg.providers.azure_responses import AzureResponsesProvider
from cyberneg.providers.base import ProviderCallContext


class _PhaseValue:
    def __init__(self, value: str):
        self.value = value


class CyberAgent:
    def __init__(
        self,
        initial_prompt_cls: Any,
        round_prompt_cls: Any,
        agent_name: str,
        temperature: float,
        model: str,
        *,
        azure: bool = False,
        role_id: str,
        line_ids: list[str],
        label_set: Dict[str, Any],
        timeout_seconds: int = 30,
        invalid_json_attempts_per_turn: int = 0,
    ):
        self.model = model
        self.agent_name = agent_name
        self.temperature = temperature
        self.initial_prompt_cls = initial_prompt_cls
        self.initial_prompt = initial_prompt_cls.return_initial_prompt()
        self.round_prompt_cls = round_prompt_cls
        self.azure = azure
        self.role_id = role_id
        self.line_ids = line_ids
        self.label_set = label_set
        self.timeout_seconds = timeout_seconds
        self.invalid_json_attempts_per_turn = max(0, int(invalid_json_attempts_per_turn))
        self._attempts_by_turn: dict[str, int] = {}
        self.provider = None
        self.provider_kind = "mock"

        if self.model.lower().startswith("mock"):
            self.provider_kind = "mock"
        elif "claude" in self.model.lower():
            self.provider_kind = "anthropic"
            api_key = self._resolve_env("ANTHROPIC_API_KEY", "ANTHROPIC_API")
            if not api_key:
                raise ValueError("Anthropic model selected but no ANTHROPIC_API_KEY / ANTHROPIC_API found")
            self.provider = AnthropicProvider(
                provider_name="anthropic",
                model_name=self.model,
                api_key=api_key,
                base_url=self._resolve_env("ANTHROPIC_BASE_URL"),
                timeout_seconds=self.timeout_seconds,
            )
        elif azure:
            self.provider_kind = "azure_responses"
            endpoint = self._resolve_env(
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_BASE",
                "OPENAI_API_BASE",
                "OPENAI_BASE_URL",
            )
            api_key = self._resolve_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API", "OPENAI_API_KEY")
            if not endpoint or not api_key:
                raise ValueError("Azure mode requires endpoint and API key in environment variables")
            self.provider = AzureResponsesProvider(
                provider_name="azure_responses",
                model_name=self.model,
                endpoint=endpoint,
                api_key=api_key,
                api_version=self._resolve_env("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION"),
                timeout_seconds=self.timeout_seconds,
            )
        else:
            raise ValueError(
                "CyberAgent supports mock models, Anthropic models containing 'claude', or Azure Responses mode via --azure."
            )

    def _resolve_env(self, *names: str) -> Optional[str]:
        for name in names:
            val = os.getenv(name)
            if val:
                return val
        return None

    def execute_round(self, answer_history: Dict[str, Any], round_idx: int):
        slot_prompt = self.round_prompt_cls.build_slot_prompt(answer_history, round_idx)
        agent_response = self.prompt(
            "user",
            slot_prompt,
            turn_id=f"public_{round_idx}_{self.role_id}",
            phase="public",
            public_turn_index=round_idx,
            is_final_public_turn=bool(answer_history.get("_current_final_turn", False)),
        )
        return slot_prompt, agent_response

    def execute_round0(self, answer_history: Dict[str, Any]):
        slot_prompt = self.round_prompt_cls.build_round0_prompt(answer_history)
        agent_response = self.prompt(
            "user",
            slot_prompt,
            turn_id=f"round0_{self.role_id}",
            phase="round0",
            public_turn_index=None,
            is_final_public_turn=False,
        )
        return slot_prompt, agent_response

    def prompt(
        self,
        role: str,
        msg: str,
        *,
        turn_id: str,
        phase: str,
        public_turn_index: Optional[int],
        is_final_public_turn: bool,
    ) -> str:
        del role  # role is kept for interface similarity with the old project.
        full_prompt = self.initial_prompt + "\n\n" + msg
        if self.provider_kind == "mock":
            return self._mock_response(
                turn_id=turn_id,
                phase=phase,
                public_turn_index=public_turn_index,
                is_final_public_turn=is_final_public_turn,
            )

        assert self.provider is not None
        ctx = ProviderCallContext(
            provider_name=self.provider.provider_name,
            provider_kind=self.provider.provider_kind,
            model_name=self.provider.model_name,
            phase=_PhaseValue(phase),
            role_id=None,
            public_turn_index=public_turn_index,
            turn_id=turn_id,
            timeout_seconds=self.timeout_seconds,
            metadata={
                "line_ids": self.line_ids,
                "label_set_labels": self.label_set.get("labels", []),
                "is_final_public_turn": is_final_public_turn,
            },
        )
        return self.provider.generate(full_prompt, ctx).text

    def _mock_response(
        self,
        *,
        turn_id: str,
        phase: str,
        public_turn_index: Optional[int],
        is_final_public_turn: bool,
    ) -> str:
        attempt = self._attempts_by_turn.get(turn_id, 0) + 1
        self._attempts_by_turn[turn_id] = attempt
        if attempt <= self.invalid_json_attempts_per_turn:
            return '{"scratchpad": "bad", "answer": '

        labels = list(self.label_set.get("labels", []))
        preferred = "XSS_Reflected" if "XSS_Reflected" in labels else labels[0]
        alt_two = "XSS_Stored" if "XSS_Stored" in labels else (labels[1] if len(labels) > 1 else preferred)
        alt_three = "NoFinding" if "NoFinding" in labels else (labels[2] if len(labels) > 2 else preferred)
        if phase == "round0":
            severity = "Low" if self.role_id == "C" else "Medium"
        else:
            severity = "Medium" if is_final_public_turn or (public_turn_index or 0) >= 2 else ("Low" if self.role_id == "C" else "Medium")

        assessment = {
            "ranked_findings": [
                {
                    "rank": 1,
                    "label": preferred,
                    "severity": severity,
                    "confidence": 0.78,
                    "citations": self.line_ids[:2] if len(self.line_ids) >= 2 else self.line_ids[:1],
                    "rationale": "The packet shows reflected input handling and browser-rendered execution evidence.",
                },
                {
                    "rank": 2,
                    "label": alt_two,
                    "severity": "Low",
                    "confidence": 0.30,
                    "citations": self.line_ids[2:3],
                    "rationale": "A different XSS variant remains possible but is less directly supported.",
                },
                {
                    "rank": 3,
                    "label": alt_three,
                    "severity": "Info",
                    "confidence": 0.15,
                    "citations": [],
                    "rationale": "A conservative no-finding interpretation remains weaker than the reflected-XSS reading.",
                },
            ],
            "decision_summary": (
                "Reflected XSS remains the most defensible current label; severity stays conservative unless stronger "
                "impact evidence appears."
            ),
        }

        seed_material = f"{turn_id}|{self.role_id}|{attempt}"
        stable_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
        verbs = ["stabilize", "clarify", "defend", "tighten"]
        verb = verbs[stable_seed % len(verbs)]
        if phase == "round0":
            public_message = (
                "My independent starting view is that the strongest explanation is a reflected XSS finding because the "
                "packet shows unsafely reflected user input and a browser-rendered proof of execution. I want the "
                "committee to keep alternatives in mind, but at this stage I would keep the leading hypothesis tightly "
                "grounded in the cited lines and avoid stretching the impact beyond what the packet directly supports. "
                "The absence of stronger impact details means I would rather converge on a defensible medium-confidence "
                "assessment now and let the later public discussion test whether any competing label explains the same "
                "packet evidence more cleanly."
            )
        else:
            public_message = (
                f"I want to {verb} around the reflected XSS hypothesis rather than widen scope without stronger "
                "evidence. The clearest support is still the reflected parameter behavior plus the browser execution "
                "proof, so my view is that the committee should converge on a defensible finding with explicit line "
                "citations and only move severity upward if someone can tie stronger impact directly to the packet. "
                "If another label is going to displace this one, I need to see a tighter mechanism argument and at "
                "least one cited line that explains the rendered script execution better than the reflected-XSS reading."
            )

        payload = {
            "scratchpad": (
                f"<SCRATCHPAD>Private notes for {self.agent_name}: compare top hypotheses, track citation quality, "
                "and preserve separation between internal reasoning and public argument.</SCRATCHPAD>"
            ),
            "answer": (
                "<ANSWER>"
                + public_message
                + "</ANSWER>\n<ASSESSMENT>"
                + json.dumps(assessment, ensure_ascii=False)
                + "</ASSESSMENT>"
            ),
            "plan": (
                "<PLAN>Keep the top-1 stable if evidence remains strongest, ask for precise citation-backed "
                "objections, and commit clearly on the final public turn.</PLAN>"
            ),
        }
        return json.dumps(payload, ensure_ascii=False)
