from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class CyberAgent:
    def __init__(
        self,
        initial_prompt_builder: Any,
        round_prompt_cls: Any,
        agent_name: str,
        temperature: float,
        model: str,
        *,
        azure: bool = False,
        role_id: str,
    ):
        del temperature
        self.model = model
        self.agent_name = agent_name
        self.initial_prompt = initial_prompt_builder.return_initial_prompt()
        self.round_prompt_cls = round_prompt_cls
        self.azure = azure
        self.role_id = role_id
        self.client = None
        self.claude_client = None
        self.claude = "claude" in self.model.lower()

        if self.claude:
            api_key = self._resolve_env("ANTHROPIC_API_KEY", "ANTHROPIC_API")
            if not api_key:
                raise ValueError("Anthropic model selected but no ANTHROPIC_API_KEY / ANTHROPIC_API found")
            base_url = self._resolve_env("ANTHROPIC_BASE_URL")
            self.claude_client = self._build_anthropic_client(api_key=api_key, base_url=base_url)
        elif self.azure:
            endpoint = self._resolve_env(
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_BASE",
                "OPENAI_API_BASE",
                "OPENAI_BASE_URL",
            )
            api_key = self._resolve_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API", "OPENAI_API_KEY")
            if not endpoint or not api_key:
                raise ValueError("Azure mode requires endpoint and API key in environment variables")
            self.client = self._build_azure_client(endpoint=endpoint, api_key=api_key)
        else:
            raise ValueError(
                "CyberAgent supports Anthropic models containing 'claude' or Azure OpenAI via --azure."
            )

    def _resolve_env(self, *names: str) -> Optional[str]:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return None

    def _build_anthropic_client(self, *, api_key: str, base_url: Optional[str]):
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("anthropic package is required for Claude models") from exc

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return Anthropic(**kwargs)

    def _resolve_azure_api_version(self) -> str:
        requested_api_version = (
            self._resolve_env("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION") or ""
        ).strip()
        required_api_version = "2024-08-01-preview"

        def _api_version_date(version: str):
            try:
                parts = version.split("-", 3)
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                return (year, month, day)
            except Exception:
                return None

        requested_date = _api_version_date(requested_api_version) if requested_api_version else None
        required_date = _api_version_date(required_api_version)
        if required_date and requested_date and requested_date < required_date:
            return required_api_version
        return requested_api_version or required_api_version

    def _build_azure_client(self, *, endpoint: str, api_key: str):
        try:
            from openai import AzureOpenAI  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("openai package is required for Azure OpenAI models") from exc

        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=self._resolve_azure_api_version(),
        )

    def _structured_output_schema(self) -> Dict[str, Any]:
        return {
            "name": "structured_output",
            "schema": {
                "type": "object",
                "properties": {
                    "scratchpad": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": r"^\s*<SCRATCHPAD>[\s\S]+</SCRATCHPAD>\s*$",
                    },
                    "public_answer": {"type": "string", "minLength": 1},
                    "assessment": {
                        "type": "object",
                        "properties": {
                            "ranked_findings": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "rank": {"type": "integer", "enum": [1, 2, 3]},
                                        "label": {"type": "string"},
                                        "severity": {
                                            "type": "string",
                                            "enum": ["Compliance", "Info", "Low", "Medium", "High"],
                                        },
                                        "citations": {"type": "array", "items": {"type": "string"}},
                                        "rationale": {"type": ["string", "null"]},
                                    },
                                    "required": ["rank", "label", "severity", "citations"],
                                    "additionalProperties": False,
                                },
                            },
                            "decision_summary": {"type": "string", "minLength": 1},
                            "accept": {"type": "boolean"},
                            "block_reason": {"type": ["string", "null"]},
                        },
                        "required": ["ranked_findings", "decision_summary", "accept", "block_reason"],
                        "additionalProperties": False,
                    },
                    "plan": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": r"^\s*<PLAN>[\s\S]+</PLAN>\s*$",
                    },
                },
                "required": ["scratchpad", "public_answer", "assessment", "plan"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _extract_claude_tool_json(self, response: Any, *, tool_name: str) -> Optional[str]:
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            block_name = getattr(block, "name", None)
            if block_type != "tool_use" or block_name != tool_name:
                continue
            payload = getattr(block, "input", None)
            if not isinstance(payload, dict):
                continue
            expected_keys = {"scratchpad", "public_answer", "assessment", "plan"}
            if set(payload.keys()) != expected_keys:
                continue
            if not isinstance(payload.get("scratchpad"), str):
                continue
            if not isinstance(payload.get("public_answer"), str):
                continue
            if not isinstance(payload.get("plan"), str):
                continue
            if not isinstance(payload.get("assessment"), dict):
                continue
            return json.dumps(payload, ensure_ascii=False)
        return None

    def execute_round(self, answer_history: Dict[str, Any], round_idx: int):
        slot_prompt = self.round_prompt_cls.build_slot_prompt(answer_history, round_idx)
        full_prompt = self.build_full_prompt(slot_prompt)
        agent_response = self.prompt(
            slot_prompt,
            turn_id=f"public_{round_idx}_{self.role_id}",
            full_prompt=full_prompt,
        )
        return slot_prompt, full_prompt, agent_response

    def execute_round0(self, answer_history: Dict[str, Any]):
        slot_prompt = self.round_prompt_cls.build_round0_prompt(answer_history)
        full_prompt = self.build_full_prompt(slot_prompt)
        agent_response = self.prompt(
            slot_prompt,
            turn_id=f"round0_{self.role_id}",
            full_prompt=full_prompt,
        )
        return slot_prompt, full_prompt, agent_response

    def build_full_prompt(self, msg: str) -> str:
        return self.initial_prompt + "\n\n" + msg

    def prompt(self, msg: str, *, turn_id: str, full_prompt: Optional[str] = None) -> str:
        del turn_id
        if full_prompt is None:
            full_prompt = self.build_full_prompt(msg)
        json_schema = self._structured_output_schema()

        if self.claude:
            assert self.claude_client is not None
            response = self.claude_client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": full_prompt}],
                tools=[
                    {
                        "name": json_schema["name"],
                        "description": "Return only the required structured output fields.",
                        "input_schema": json_schema["schema"],
                    }
                ],
                tool_choice={"type": "tool", "name": json_schema["name"]},
            )
            tool_json = self._extract_claude_tool_json(response, tool_name=json_schema["name"])
            if tool_json is not None:
                return tool_json

            parts: list[str] = []
            for block in getattr(response, "content", []) or []:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)

        assert self.client is not None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            response_format={"type": "json_schema", "json_schema": json_schema},
        )
        content = response.choices[0].message.content
        return content or ""
