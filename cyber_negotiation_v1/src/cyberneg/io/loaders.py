from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from ..core.schemas import (
    ConditionSetConfig,
    EvidencePacket,
    ExperimentConfig,
    GroundTruth,
    LabelSetConfig,
    ModelCatalog,
    PromptTemplateConfig,
    ProviderCatalog,
    RoleInstructionConfig,
)
from ..providers.anthropic import AnthropicProvider
from ..providers.azure_responses import AzureResponsesProvider
from ..providers.base import BaseProvider
from ..providers.mock_provider import MockProvider


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_env_file_if_present(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_yaml_model(path: Path, model_cls):
    data = yaml.safe_load(_read_text(path))
    return model_cls.model_validate(data)


def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start.resolve().parent if start.is_file() else start.resolve()


def resolve_path(raw: str, *, config_path: Path, project_root: Path) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((config_path.parent / p).resolve())
        candidates.append((project_root / p).resolve())
    for c in candidates:
        if c.exists():
            return c
    # Return project-root relative default candidate for clearer errors.
    return (project_root / p).resolve() if not p.is_absolute() else p


@dataclass
class LoadedExperimentBundle:
    project_root: Path
    experiment_config_path: Path
    experiment: ExperimentConfig
    condition_set: ConditionSetConfig
    provider_catalog: ProviderCatalog
    model_catalog: ModelCatalog
    roles: dict[str, RoleInstructionConfig]
    prompts: dict[str, PromptTemplateConfig]
    label_sets: dict[str, LabelSetConfig]
    scenarios: dict[str, EvidencePacket]
    ground_truths: dict[str, GroundTruth]
    config_refs: dict[str, str]


def load_experiment_bundle(config_path: str | Path) -> LoadedExperimentBundle:
    cfg_path = Path(config_path).resolve()
    project_root = _find_project_root(cfg_path)
    experiment = load_yaml_model(cfg_path, ExperimentConfig)

    cond_path = resolve_path(experiment.condition_set_path, config_path=cfg_path, project_root=project_root)
    prov_path = resolve_path(experiment.provider_catalog_path, config_path=cfg_path, project_root=project_root)
    model_path = resolve_path(experiment.model_catalog_path, config_path=cfg_path, project_root=project_root)
    condition_set = load_yaml_model(cond_path, ConditionSetConfig)
    provider_catalog = load_yaml_model(prov_path, ProviderCatalog)
    model_catalog = load_yaml_model(model_path, ModelCatalog)

    roles: dict[str, RoleInstructionConfig] = {}
    for role_id, role_path_raw in experiment.role_paths.items():
        role_path = resolve_path(role_path_raw, config_path=cfg_path, project_root=project_root)
        roles[role_id.value] = load_yaml_model(role_path, RoleInstructionConfig)

    prompts: dict[str, PromptTemplateConfig] = {}
    for key, prompt_path_raw in experiment.prompt_paths.items():
        prompt_path = resolve_path(prompt_path_raw, config_path=cfg_path, project_root=project_root)
        prompts[key] = load_yaml_model(prompt_path, PromptTemplateConfig)

    label_sets: dict[str, LabelSetConfig] = {}
    for raw in experiment.label_set_paths:
        p = resolve_path(raw, config_path=cfg_path, project_root=project_root)
        label_set = load_yaml_model(p, LabelSetConfig)
        label_sets[label_set.label_set_id] = label_set

    scenarios: dict[str, EvidencePacket] = {}
    for raw in experiment.scenario_paths:
        p = resolve_path(raw, config_path=cfg_path, project_root=project_root)
        scenario = load_yaml_model(p, EvidencePacket)
        scenarios[scenario.scenario_id] = scenario

    ground_truths: dict[str, GroundTruth] = {}
    for raw in experiment.ground_truth_paths:
        p = resolve_path(raw, config_path=cfg_path, project_root=project_root)
        gt = load_yaml_model(p, GroundTruth)
        ground_truths[gt.scenario_id] = gt

    return LoadedExperimentBundle(
        project_root=project_root,
        experiment_config_path=cfg_path,
        experiment=experiment,
        condition_set=condition_set,
        provider_catalog=provider_catalog,
        model_catalog=model_catalog,
        roles=roles,
        prompts=prompts,
        label_sets=label_sets,
        scenarios=scenarios,
        ground_truths=ground_truths,
        config_refs={
            "experiment_config": str(cfg_path),
            "condition_set": str(cond_path),
            "provider_catalog": str(prov_path),
            "model_catalog": str(model_path),
        },
    )


def resolve_env_alias(env_mapping: dict[str, list[str]], key: str) -> Optional[str]:
    for env_name in env_mapping.get(key, []):
        val = os.getenv(env_name)
        if val:
            return val
    return None


def build_provider_from_model_ref(
    *,
    bundle: LoadedExperimentBundle,
    model_ref: str,
    timeout_override: Optional[int] = None,
    mock_behavior_override: Optional[dict[str, Any]] = None,
) -> BaseProvider:
    if model_ref not in bundle.model_catalog.models:
        raise KeyError(f"Unknown model ref '{model_ref}'")
    model_spec = bundle.model_catalog.models[model_ref]
    provider_name = model_spec.provider
    if provider_name not in bundle.provider_catalog.providers:
        raise KeyError(f"Unknown provider '{provider_name}' referenced by model '{model_ref}'")
    provider_spec = bundle.provider_catalog.providers[provider_name]

    timeout_seconds = timeout_override if timeout_override is not None else model_spec.timeout_seconds

    if provider_spec.kind == "mock":
        mb = mock_behavior_override or {}
        return MockProvider(
            provider_name=provider_name,
            model_name=model_spec.model_name,
            invalid_json_attempts_per_turn=int(mb.get("invalid_json_attempts_per_turn", 0)),
            deterministic_seed_offset=int(mb.get("deterministic_seed_offset", 0)),
        )

    if provider_spec.kind == "azure_responses":
        endpoint = resolve_env_alias(provider_spec.env_mapping, "endpoint")
        api_key = resolve_env_alias(provider_spec.env_mapping, "api_key")
        api_version = resolve_env_alias(provider_spec.env_mapping, "api_version")
        if not endpoint or not api_key:
            raise RuntimeError("Azure Responses provider requires endpoint and api_key via configured env aliases")
        return AzureResponsesProvider(
            provider_name=provider_name,
            model_name=model_spec.model_name,
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
        )

    if provider_spec.kind == "anthropic":
        api_key = resolve_env_alias(provider_spec.env_mapping, "api_key")
        base_url = resolve_env_alias(provider_spec.env_mapping, "base_url")
        if not api_key:
            raise RuntimeError("Anthropic provider requires api_key via configured env aliases")
        return AnthropicProvider(
            provider_name=provider_name,
            model_name=model_spec.model_name,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    raise RuntimeError(f"Unsupported provider kind '{provider_spec.kind}'")


def dump_bundle_summary(bundle: LoadedExperimentBundle) -> dict[str, Any]:
    return {
        "experiment_id": bundle.experiment.experiment_id,
        "conditions": [c.condition_id for c in bundle.condition_set.conditions],
        "scenarios": list(bundle.scenarios.keys()),
        "label_sets": list(bundle.label_sets.keys()),
        "roles": list(bundle.roles.keys()),
    }

