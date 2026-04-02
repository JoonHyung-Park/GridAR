import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import trange

from data_utils import load_sample_records, resolve_benchmark_root, slice_sample_records, validate_benchmark_config
from method_utils import build_t2i_request, load_generation_runtime, load_orm_model, run_orm_ranking, tokenize_generation_prompts
from utils import OutputPaths, prepare_output_paths, resolve_eval_paths, resolve_partition_range, sanitize_component, save_config_snapshot, set_seed

def build_method_name(cfg: DictConfig, model_name: str) -> str:
    components = [
        cfg.method.name,
        model_name,
        f"{cfg.orm.model_name}ORM",
        cfg.benchmark.name,
        f"cfg{cfg.cfg_scale}",
        f"N{cfg.N}",
    ]
    return "__".join(sanitize_component(component) for component in components)


@hydra.main(version_base=None, config_path="configs", config_name="evaluate_bon")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    benchmark_adapter = validate_benchmark_config(cfg)

    cfg.model_params.model.path = cfg.model_params.model.path or os.environ.get("MODEL_PATH") or os.environ.get("JANUS_MODEL_PATH")
    cfg.orm.model_path = cfg.orm.model_path or os.environ.get("ORM_MODEL_PATH")
    if not cfg.orm.model_path:
        raise ValueError("Set orm.model_path.")

    script_path = Path(__file__)
    repo_root = script_path.resolve().parent.parent
    benchmark_root = resolve_benchmark_root(cfg, repo_root, benchmark_adapter)
    paths = resolve_eval_paths(cfg, script_path, benchmark_root)

    runtime = load_generation_runtime(cfg, paths.model_path)
    model_name = runtime.model_name
    orm_model, orm_processor = load_orm_model(paths.orm_model_path)

    method_name = build_method_name(cfg, model_name)
    output_paths: OutputPaths = prepare_output_paths(paths.output_root, model_name, method_name)

    sample_records = load_sample_records(cfg, paths.dataset_path, benchmark_adapter)
    total_samples = len(sample_records)
    data_start_idx, data_end_idx = resolve_partition_range(total_samples, cfg.batch_size, cfg.partition_id, cfg.total_partitions)

    for start_idx in trange(data_start_idx, data_end_idx, cfg.batch_size):
        end_idx = min(start_idx + cfg.batch_size, data_end_idx)
        batch_records = slice_sample_records(sample_records, start_idx, end_idx, benchmark_adapter)
        if not batch_records:
            print(f"Batch {start_idx}-{end_idx - 1} contains only duplicate prompts. Skipping.")
            continue
        benchmark_adapter.prepare_batch_outputs(batch_records, output_paths.sample_dir)
        batch_prompts = [record.prompt for record in batch_records]
        batch_names = [record.save_name for record in batch_records]

        save_paths = [output_paths.sample_dir / name for name in batch_names]
        if all(path.exists() for path in save_paths):
            print(f"Batch {start_idx}-{end_idx - 1} is already generated.")
            continue

        batch_size = len(batch_prompts)
        set_seed(start_idx)

        expanded_prompts = [batch_prompts[index // cfg.N] for index in range(batch_size * cfg.N)]
        input_embeds, attention_mask = tokenize_generation_prompts(
            runtime,
            expanded_prompts,
            None,
            use_3way_cfg=False,
        )

        stage_start_time = time.time()
        request = build_t2i_request(cfg, input_embeds, attention_mask)
        # Keep the first-quarter aligned with GridAR before selection/refinement.
        final_tokens = runtime.generator.generate_t2i_stage_prefix_preserving(request)
        final_images = runtime.generator.image_decode(request, final_tokens).astype("uint8")

        torch.save(final_tokens.cpu(), output_paths.records_dir / f"batch{start_idx:04d}-{end_idx - 1}_tokens.pt")

        images_bon = final_images.reshape(batch_size, cfg.N, cfg.model_params.img_size, cfg.model_params.img_size, 3)
        bon_scores = run_orm_ranking(
            batch_prompts=batch_prompts,
            images_bon=images_bon,
            orm_model=orm_model,
            orm_processor=orm_processor,
        )

        selected_indices = bon_scores.argmax(dim=1)
        for batch_idx, save_path in enumerate(save_paths):
            if save_path.exists():
                continue
            selected = int(selected_indices[batch_idx].item())
            final_image = Image.fromarray(images_bon[batch_idx, selected].astype("uint8"))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            final_image.save(save_path)

        elapsed = time.time() - stage_start_time
        print(f"Completed batch {start_idx}-{end_idx - 1} in {elapsed:.2f}s")

    save_config_snapshot(
        cfg,
        output_paths,
        stats={},
        extras={
            "model_path": str(paths.model_path),
            "orm_model_path": str(paths.orm_model_path),
            "benchmark_root": str(paths.benchmark_root),
        },
    )


if __name__ == "__main__":
    main()
