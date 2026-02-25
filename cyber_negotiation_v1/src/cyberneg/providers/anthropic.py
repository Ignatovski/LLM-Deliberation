from __future__ import annotations

from typing import Optional

from .base import BaseProvider, ProviderCallContext, ProviderResponse


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        provider_name: str,
        model_name: str,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        super().__init__(provider_name=provider_name, provider_kind="anthropic", model_name=model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._client = None

    def _client_instance(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("anthropic package is required for AnthropicProvider") from exc
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout_seconds is not None:
            kwargs["timeout"] = self.timeout_seconds
        self._client = Anthropic(**kwargs)
        return self._client

    def generate(self, prompt: str, ctx: ProviderCallContext) -> ProviderResponse:
        client = self._client_instance()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        text = "".join(parts)
        usage = getattr(response, "usage", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": None,
            }
        return ProviderResponse(
            text=text,
            usage=usage_dict,
            request_id=getattr(response, "id", None),
            request_metadata={
                "phase": ctx.phase.value,
                "role_id": ctx.role_id.value if ctx.role_id else None,
                "public_turn_index": ctx.public_turn_index,
            },
            response_metadata={"provider_kind": self.provider_kind},
        )

