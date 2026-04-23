from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
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
        self.azure_runtime: Optional[Dict[str, Any]] = None
        self.claude_runtime: Optional[Dict[str, Any]] = None
        self.claude = "claude" in self.model.lower()

        if self.claude:
            api_key = self._resolve_env("ANTHROPIC_API_KEY", "ANTHROPIC_API")
            if not api_key:
                raise ValueError("Anthropic model selected but no ANTHROPIC_API_KEY / ANTHROPIC_API found")
            base_url = self._resolve_env("ANTHROPIC_BASE_URL")
            self.claude_runtime = self._build_anthropic_runtime(api_key=api_key, base_url=base_url)
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
            self.azure_runtime = self._build_azure_runtime(endpoint=endpoint, api_key=api_key)
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

    def _resolve_claude_max_tokens(self) -> int:
        raw_value = (
            self._resolve_env("ANTHROPIC_MAX_TOKENS", "CLAUDE_MAX_TOKENS", "OPENAI_MAX_COMPLETION_TOKENS") or ""
        ).strip()
        if raw_value:
            try:
                parsed = int(raw_value)
                if parsed > 0:
                    return parsed
            except Exception:
                pass
        return 2048

    def _build_anthropic_runtime(self, *, api_key: str, base_url: Optional[str]) -> Dict[str, Any]:
        timeout_raw = (
            self._resolve_env("ANTHROPIC_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS") or "90"
        ).strip()
        retries_raw = (
            self._resolve_env("ANTHROPIC_MAX_RETRIES", "OPENAI_MAX_RETRIES") or "0"
        ).strip()
        try:
            timeout_seconds = float(timeout_raw)
        except Exception:
            timeout_seconds = 90.0
        try:
            max_retries = int(retries_raw)
        except Exception:
            max_retries = 0

        return {
            "api_key": api_key,
            "base_url": (base_url or "https://api.anthropic.com").strip(),
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }

    def _build_anthropic_messages_url(self) -> str:
        assert self.claude_runtime is not None
        base_url = str(self.claude_runtime["base_url"]).rstrip("/")
        if base_url.lower().endswith("/v1/messages"):
            return base_url
        return base_url + "/v1/messages"

    def _anthropic_request_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self.claude_runtime is not None
        url = self._build_anthropic_messages_url()
        timeout_seconds = float(self.claude_runtime["timeout"])
        max_retries = int(self.claude_runtime["max_retries"])
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_exc: Optional[BaseException] = None

        for attempt in range(max_retries + 1):
            request = urllib.request.Request(url, data=body, method="POST")
            request.add_header("x-api-key", str(self.claude_runtime["api_key"]))
            request.add_header("anthropic-version", "2023-06-01")
            request.add_header("content-type", "application/json")
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                try:
                    return json.loads(raw)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError("Anthropic returned non-JSON response") from exc
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Anthropic HTTP {exc.code}: {detail[:1000]}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break

        raise TimeoutError(f"Anthropic request timed out after {timeout_seconds:.0f}s") from last_exc

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

    def _build_azure_runtime(self, *, endpoint: str, api_key: str) -> Dict[str, Any]:
        timeout_raw = (
            self._resolve_env("AZURE_OPENAI_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS") or "90"
        ).strip()
        retries_raw = (
            self._resolve_env("AZURE_OPENAI_MAX_RETRIES", "OPENAI_MAX_RETRIES") or "0"
        ).strip()
        try:
            timeout_seconds = float(timeout_raw)
        except Exception:
            timeout_seconds = 90.0
        try:
            max_retries = int(retries_raw)
        except Exception:
            max_retries = 0

        return {
            "endpoint": endpoint.strip(),
            "api_key": api_key,
            "api_version": self._resolve_azure_api_version(),
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "openai_compatible": self._is_openai_compatible_endpoint(endpoint),
        }

    def _is_openai_compatible_endpoint(self, endpoint: str) -> bool:
        normalized = str(endpoint or "").strip().rstrip("/").lower()
        return normalized.endswith("/openai/v1") or "/openai/v1/" in normalized

    def _build_azure_chat_url(self, model: str) -> str:
        assert self.azure_runtime is not None
        endpoint = str(self.azure_runtime["endpoint"]).rstrip("/")
        quoted_model = urllib.parse.quote(model, safe="")
        lowered_endpoint = endpoint.lower()
        if bool(self.azure_runtime.get("openai_compatible")):
            if lowered_endpoint.endswith("/chat/completions"):
                return endpoint
            return endpoint + "/chat/completions"
        if lowered_endpoint.endswith("/chat/completions"):
            url = endpoint
        elif "/openai/deployments/" in lowered_endpoint:
            url = endpoint + "/chat/completions"
        else:
            url = endpoint + f"/openai/deployments/{quoted_model}/chat/completions"
        if "api-version=" not in url:
            separator = "&" if "?" in url else "?"
            url += f"{separator}api-version={urllib.parse.quote(str(self.azure_runtime['api_version']), safe='')}"
        return url

    def _azure_request_json(self, *, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self.azure_runtime is not None
        url = self._build_azure_chat_url(model)
        timeout_seconds = float(self.azure_runtime["timeout_seconds"])
        max_retries = int(self.azure_runtime["max_retries"])
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_exc: Optional[BaseException] = None

        for attempt in range(max_retries + 1):
            request = urllib.request.Request(url, data=body, method="POST")
            if bool(self.azure_runtime.get("openai_compatible")):
                request.add_header("Authorization", f"Bearer {self.azure_runtime['api_key']}")
            else:
                request.add_header("api-key", str(self.azure_runtime["api_key"]))
            request.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                try:
                    return json.loads(raw)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError("Azure returned non-JSON response") from exc
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Azure HTTP {exc.code}: {detail[:1000]}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break

        raise TimeoutError(f"Azure request timed out after {timeout_seconds:.0f}s") from last_exc

    def _resolve_max_completion_tokens(self) -> Optional[int]:
        raw_value = (
            self._resolve_env("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "OPENAI_MAX_COMPLETION_TOKENS") or ""
        ).strip()
        if raw_value:
            try:
                parsed = int(raw_value)
                if parsed > 0:
                    return parsed
            except Exception:
                return None
        if "gpt-5" in self.model.lower():
            return 1024
        return None

    def _azure_chat_completion(self, request_kwargs: Dict[str, Any]) -> str:
        model = str(request_kwargs["model"])
        token_cap = self._resolve_max_completion_tokens()
        caps_to_try: list[Optional[int]]
        if "gpt-5" in model.lower() and (
            self._resolve_env("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "OPENAI_MAX_COMPLETION_TOKENS") or ""
        ).strip() == "":
            caps_to_try = [token_cap or 1024, max((token_cap or 1024) * 2, 1536)]
        else:
            caps_to_try = [token_cap]

        for idx, cap in enumerate(caps_to_try):
            payload = dict(request_kwargs)
            if cap is not None:
                payload["max_completion_tokens"] = cap
            response = self._azure_request_json(model=model, payload=payload)
            choices = response.get("choices") or []
            if not choices:
                return ""
            first_choice = choices[0] or {}
            message = first_choice.get("message") or {}
            content = message.get("content") or ""
            finish_reason = str(first_choice.get("finish_reason") or "")
            if content or finish_reason != "length" or idx == len(caps_to_try) - 1:
                return str(content)
        return ""

    def _structured_output_schema(self) -> Dict[str, Any]:
        return {
            "name": "structured_output",
            "schema": {
                "type": "object",
                "properties": {
                    "scratchpad": {"type": "string", "minLength": 1},
                    "public_answer": {"type": "string"},
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
                                    "required": ["rank", "label", "severity", "citations", "rationale"],
                                    "additionalProperties": False,
                                },
                            },
                            "decision_summary": {"type": "string"},
                            "accept": {"type": "boolean"},
                            "block_reason": {"type": ["string", "null"]},
                            "user_assumption_verdict": {
                                "type": "string",
                                "enum": ["Correct", "Wrong", "NoAssumption"],
                            },
                            "user_assumption_statement": {"type": "string"},
                        },
                        "required": [
                            "ranked_findings",
                            "decision_summary",
                            "accept",
                            "block_reason",
                            "user_assumption_verdict",
                            "user_assumption_statement",
                        ],
                        "additionalProperties": False,
                    },
                    "plan": {"type": "string", "minLength": 1},
                },
                "required": ["scratchpad", "public_answer", "assessment", "plan"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _extract_claude_tool_payload(self, response: Any, *, tool_name: str) -> Optional[Dict[str, Any]]:
        content_blocks = response.get("content") if isinstance(response, dict) else getattr(response, "content", [])
        for block in content_blocks or []:
            if isinstance(block, dict):
                block_type = block.get("type")
                block_name = block.get("name")
                payload = block.get("input")
            else:
                block_type = getattr(block, "type", None)
                block_name = getattr(block, "name", None)
                payload = getattr(block, "input", None)
            if block_type != "tool_use" or block_name != tool_name:
                continue
            if not isinstance(payload, dict):
                continue
            return payload
        return None

    def _extract_claude_tool_json(self, response: Any, *, tool_name: str) -> Optional[str]:
        payload = self._extract_claude_tool_payload(response, tool_name=tool_name)
        if not isinstance(payload, dict):
            return None
        expected_keys = {"scratchpad", "public_answer", "assessment", "plan"}
        if set(payload.keys()) != expected_keys:
            return None
        if not isinstance(payload.get("scratchpad"), str):
            return None
        if not isinstance(payload.get("public_answer"), str):
            return None
        if not isinstance(payload.get("plan"), str):
            return None
        if not isinstance(payload.get("assessment"), dict):
            return None
        return json.dumps(payload, ensure_ascii=False)

    def _extract_claude_text(self, response: Any) -> str:
        parts: list[str] = []
        content_blocks = response.get("content") if isinstance(response, dict) else getattr(response, "content", [])
        for block in content_blocks or []:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    def _claude_tool_output_is_incomplete(self, response: Any, *, tool_name: str) -> bool:
        payload = self._extract_claude_tool_payload(response, tool_name=tool_name)
        if not isinstance(payload, dict):
            return False
        expected_keys = {"scratchpad", "public_answer", "assessment", "plan"}
        return set(payload.keys()) != expected_keys

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
            base_cap = self._resolve_claude_max_tokens()
            caps_to_try = [base_cap, max(base_cap + 1024, int(base_cap * 1.5)), max(base_cap + 2048, base_cap * 2)]
            last_response: Optional[Dict[str, Any]] = None
            for idx, cap in enumerate(caps_to_try):
                response = self._anthropic_request_json(
                    {
                        "model": self.model,
                        "max_tokens": cap,
                        "messages": [{"role": "user", "content": full_prompt}],
                        "tools": [
                            {
                                "name": json_schema["name"],
                                "description": "Return only the required structured output fields.",
                                "input_schema": json_schema["schema"],
                            }
                        ],
                        "tool_choice": {"type": "tool", "name": json_schema["name"]},
                    }
                )
                last_response = response
                tool_json = self._extract_claude_tool_json(response, tool_name=json_schema["name"])
                if tool_json is not None:
                    return tool_json
                stop_reason = ""
                if isinstance(response, dict):
                    stop_reason = str(response.get("stop_reason") or "")
                if (
                    self._claude_tool_output_is_incomplete(response, tool_name=json_schema["name"])
                    and stop_reason == "max_tokens"
                    and idx < len(caps_to_try) - 1
                ):
                    continue
                text_output = self._extract_claude_text(response)
                if text_output or idx == len(caps_to_try) - 1:
                    return text_output
            if last_response is not None:
                return self._extract_claude_text(last_response)
            return ""

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "response_format": {"type": "json_schema", "json_schema": json_schema},
        }

        if "gpt-5" in self.model.lower():
            reasoning_effort = (
                self._resolve_env("AZURE_OPENAI_REASONING_EFFORT", "OPENAI_REASONING_EFFORT") or "minimal"
            ).strip()
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort

        return self._azure_chat_completion(request_kwargs)
