import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import set_seed as hf_set_seed


_SECRET_KEYS = {"api_key", "token", "secret", "password"}


@dataclass(frozen=True)
class EvalPaths:
    repo_root: Path
    model_path: Path
    benchmark_root: Path
    dataset_path: Path
    output_root: Path
    orm_model_path: Path


@dataclass(frozen=True)
class OutputPaths:
    sample_dir: Path
    records_dir: Path
    config_dir: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)


def resolve_path(path_value: Optional[str], repo_root: Path, default_relpath: Optional[str] = None) -> Path:
    raw_value = path_value or default_relpath
    if raw_value is None:
        raise ValueError("A path value is required but none was provided.")

    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def resolve_eval_paths(cfg, script_path: Path, benchmark_root: Path) -> EvalPaths:
    repo_root = script_path.resolve().parent.parent

    if cfg.model_params.model.path is None:
        raise ValueError("Set model_params.model.path.")
    model_path = resolve_path(cfg.model_params.model.path, repo_root)

    if cfg.orm.model_path is None:
        raise ValueError("Set orm.model_path.")
    orm_model_path = resolve_path(cfg.orm.model_path, repo_root)

    dataset_path = resolve_path(cfg.benchmark.dataset_file, benchmark_root)
    output_root = resolve_path(
        cfg.output_root or None,
        repo_root,
        default_relpath=getattr(cfg.benchmark, "default_output_root", None),
    )

    return EvalPaths(
        repo_root=repo_root,
        model_path=model_path,
        benchmark_root=benchmark_root,
        dataset_path=dataset_path,
        output_root=output_root,
        orm_model_path=orm_model_path,
    )


def prepare_output_paths(output_root: Path, model_name: str, method_name: str) -> OutputPaths:
    sample_dir = output_root / "generated" / model_name / method_name / "samples"
    records_dir = output_root / "generated" / model_name / method_name / "records"
    config_dir = output_root / "generated" / model_name / method_name / "config"

    for path in (sample_dir, records_dir, config_dir):
        path.mkdir(parents=True, exist_ok=True)

    return OutputPaths(sample_dir=sample_dir, records_dir=records_dir, config_dir=config_dir)


def resolve_partition_range(total_samples: int, batch_size: int, partition_id: int, total_partitions: int) -> tuple[int, int]:
    if total_partitions <= 0:
        raise ValueError("total_partitions must be positive.")
    if not (0 <= partition_id < total_partitions):
        raise ValueError("partition_id must satisfy 0 <= partition_id < total_partitions.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    total_steps = math.ceil(total_samples / batch_size)
    step_start = math.ceil(total_steps * partition_id / total_partitions)
    step_end = math.ceil(total_steps * (partition_id + 1) / total_partitions)
    return min(total_samples, step_start * batch_size), min(total_samples, step_end * batch_size)


def save_config_snapshot(cfg, output_paths: OutputPaths, stats: dict[str, Any], extras: Optional[dict[str, Any]] = None) -> None:
    payload = {"cfg": redact_secrets(OmegaConf.to_container(cfg, resolve=True)), **stats}
    if extras:
        payload.update(extras)

    config_path = output_paths.config_dir / f"config_{cfg.partition_id}_over_{cfg.total_partitions}.json"
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def is_secret_key_name(key: str) -> bool:
    if key in _SECRET_KEYS:
        return True

    return any(key.endswith(f"_{secret_key}") for secret_key in _SECRET_KEYS)


def redact_secrets(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted = {}
        for key, value in payload.items():
            lowered = key.lower()
            if is_secret_key_name(lowered):
                redacted[key] = "***"
            else:
                redacted[key] = redact_secrets(value)
        return redacted

    if isinstance(payload, list):
        return [redact_secrets(item) for item in payload]

    return payload
