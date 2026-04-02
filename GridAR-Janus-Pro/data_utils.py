import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig

from utils import resolve_path


@dataclass(frozen=True)
class SampleRecord:
    prompt: str
    save_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkAdapter:
    family: str
    validate_config: Callable[[DictConfig], None]
    load_records: Callable[[DictConfig, Path], list[SampleRecord]]
    slice_batch: Callable[[list[SampleRecord], int, int], list[SampleRecord]]
    prepare_batch_outputs: Callable[[list[SampleRecord], Path], None]


def infer_benchmark_family(cfg: DictConfig) -> str:
    if getattr(cfg.benchmark, "family", None) is not None:
        return str(cfg.benchmark.family).lower()

    benchmark_name = str(cfg.benchmark.name).lower()
    if "compbench" in benchmark_name:
        return "compbench"

    raise NotImplementedError(f"Unsupported benchmark: {cfg.benchmark.name}")


def load_nonempty_lines(text_file: Path) -> list[str]:
    return [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_compbench_records(cfg: DictConfig, dataset_path: Path) -> list[SampleRecord]:
    prompts = load_nonempty_lines(dataset_path)
    sample_records: list[SampleRecord] = []
    question_id = 0

    for dataset_index, prompt in enumerate(prompts):
        prompt_stem = prompt.replace("/", "_")
        for repeat_index in range(cfg.benchmark.repeat):
            sample_records.append(
                SampleRecord(
                    prompt=prompt,
                    save_name=f"{prompt_stem}_{question_id:06d}.png",
                    metadata={
                        "dataset_index": dataset_index,
                        "repeat_index": repeat_index,
                        "question_id": question_id,
                    },
                )
            )
            question_id += 1

    return sample_records


def validate_compbench_config(cfg: DictConfig) -> None:
    if cfg.benchmark.repeat != 2:
        raise ValueError("CompBench requires benchmark.repeat=2.")


def prepare_noop_batch_outputs(batch_records: list[SampleRecord], sample_dir: Path) -> None:
    return None


def slice_standard_batch(sample_records: list[SampleRecord], start_idx: int, end_idx: int) -> list[SampleRecord]:
    return sample_records[start_idx:end_idx]


def slice_compbench_batch(sample_records: list[SampleRecord], start_idx: int, end_idx: int) -> list[SampleRecord]:
    batch_records = sample_records[start_idx:end_idx]
    seen_samples = {
        (record.prompt, int(record.metadata["repeat_index"]))
        for record in sample_records[:start_idx]
    }
    remaining_records: list[SampleRecord] = []
    for record in batch_records:
        sample_key = (record.prompt, int(record.metadata["repeat_index"]))
        if sample_key in seen_samples:
            continue
        remaining_records.append(record)
        seen_samples.add(sample_key)
    return remaining_records


BENCHMARK_ADAPTERS = {
    "compbench": BenchmarkAdapter(
        family="compbench",
        validate_config=validate_compbench_config,
        load_records=load_compbench_records,
        slice_batch=slice_compbench_batch,
        prepare_batch_outputs=prepare_noop_batch_outputs,
    ),
}


def validate_benchmark_config(cfg: DictConfig) -> BenchmarkAdapter:
    benchmark_family = infer_benchmark_family(cfg)
    if benchmark_family not in BENCHMARK_ADAPTERS:
        raise NotImplementedError(f"Unsupported benchmark: {cfg.benchmark.name}")

    adapter = BENCHMARK_ADAPTERS[benchmark_family]
    adapter.validate_config(cfg)
    return adapter


def load_sample_records(cfg: DictConfig, dataset_path: Path, adapter: BenchmarkAdapter | None = None) -> list[SampleRecord]:
    resolved_adapter = adapter or validate_benchmark_config(cfg)
    return resolved_adapter.load_records(cfg, dataset_path)


def slice_sample_records(
    sample_records: list[SampleRecord],
    start_idx: int,
    end_idx: int,
    adapter: BenchmarkAdapter,
) -> list[SampleRecord]:
    return adapter.slice_batch(sample_records, start_idx, end_idx)


def resolve_benchmark_root(cfg: DictConfig, repo_root: Path, adapter: BenchmarkAdapter) -> Path:
    benchmark_root_override = getattr(cfg, "benchmark_root", None) or os.environ.get("BENCHMARK_ROOT")
    return resolve_path(
        benchmark_root_override,
        repo_root,
        default_relpath=getattr(cfg.benchmark, "default_root", f"benchmark/{adapter.family}"),
    )
