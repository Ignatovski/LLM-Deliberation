from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    COMPLIANCE = "Compliance"
    INFO = "Info"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.COMPLIANCE: 0,
    Severity.INFO: 1,
    Severity.LOW: 2,
    Severity.MEDIUM: 3,
    Severity.HIGH: 4,
}


class RoleId(str, Enum):
    R = "R"
    C = "C"
    K = "K"


ROLE_ORDER: tuple[RoleId, RoleId, RoleId] = (RoleId.R, RoleId.C, RoleId.K)


class ConditionMode(str, Enum):
    NEGOTIATION = "negotiation"
    BASELINE = "baseline"


class TurnPhase(str, Enum):
    ROUND0 = "round0"
    PUBLIC = "public"
    BASELINE = "baseline"


class TurnStatus(str, Enum):
    SUCCESS = "success"
    FAILED_JSON = "failed_json"
    FAILED_SCHEMA = "failed_schema"
    FAILED_RUNTIME = "failed_runtime"

