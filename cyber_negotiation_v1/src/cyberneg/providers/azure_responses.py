from __future__ import annotations

import os
from typing import Any, Optional

from .base import BaseProvider, ProviderCallContext, ProviderResponse


class AzureResponsesProvider(BaseProvider):
    """
    Azure OpenAI Responses API adapter scaffold.

    Note:
    - This is wired for V1 config/runtime integration, but exact payload details may
      need environment-specific adjustment (documented in docs/open_questions.md).
    """

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        *,
        endpoint: str,
        api_key: str,
        api_version: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        super().__init__(provider_name=provider_name, provider_kind="azure_responses", model_name=model_name)
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        self.timeout_seconds = timeout_seconds
        self._client = None

    def _client_instance(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AzureOpenAI  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("openai package is required for AzureResponsesProvider") from exc
        self._client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            timeout=self.timeout_seconds,
        )
        return self._client

    def generate(self, prompt: str, ctx: ProviderCallContext) -> ProviderResponse:
        client = self._client_instance()
        # Conservative Responses API call scaffold. If a specific SDK version differs,
        # the raised exception is logged by the strict retry loop.
        response = client.responses.create(  # type: ignore[attr-defined]
            model=self.model_name,
            input=prompt,
        )
        text = getattr(response, "output_text", None)
        if not isinstance(text, str):
            # Best-effort extraction without fallback parsing of semantics; only transport extraction.
            text = str(response)
        usage = getattr(response, "usage", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "raw": getattr(usage, "model_dump", lambda: None)() if hasattr(usage, "model_dump") else None,
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

