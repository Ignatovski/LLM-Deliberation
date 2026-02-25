from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.enums import RoleId, TurnPhase


@dataclass
class ProviderCallContext:
    provider_name: str
    provider_kind: str
    model_name: str
    phase: TurnPhase
    role_id: Optional[RoleId]
    public_turn_index: Optional[int]
    turn_id: str
    timeout_seconds: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    text: str
    usage: Optional[dict[str, Any]] = None
    request_id: Optional[str] = None
    request_metadata: Optional[dict[str, Any]] = None
    response_metadata: Optional[dict[str, Any]] = None


class BaseProvider(ABC):
    def __init__(self, provider_name: str, provider_kind: str, model_name: str):
        self.provider_name = provider_name
        self.provider_kind = provider_kind
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, ctx: ProviderCallContext) -> ProviderResponse:
        raise NotImplementedError

