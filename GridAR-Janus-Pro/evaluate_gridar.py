import os
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import trange

from data_utils import load_sample_records, resolve_benchmark_root, slice_sample_records, validate_benchmark_config
from method_utils import build_candidate_names, build_group_names, build_t2i_request, build_visible_stage_images, load_generation_runtime, load_orm_model, load_prm_client, load_prm_model, refine_stage_prompts, run_orm_ranking, run_stage_selection, save_intermediate_outputs, save_text_records, tokenize_generation_prompts
from utils import OutputPaths, prepare_output_paths, resolve_eval_paths, resolve_partition_range, sanitize_component, save_config_snapshot, set_seed


GPT_SELECT_FIRST_QUARTERS_PROMPT = """You are given a single image consisting of 4 contiguous horizontal quarters (from top to bottom: quarter 1, quarter 2, quarter 3, quarter 4).
Each quarter shows the top quarter (upper 1/4 crop) of a different full image generated from the same text prompt. The lower three-quarters of each full image are not shown in this composite. Since only the top part is visible, some quarters may show only background without any objects, while in other cases objects may appear only partially, with the rest continuing into the unseen lower part of the image.

The text prompt is: "{}".

For each of the 4 quarters, answer strictly with either "possible" or "impossible" (lowercase, no punctuation).
Output must contain exactly 4 words, separated by commas, in order: quarter 1, quarter 2, quarter 3, and quarter 4.
Example format: possible, impossible, possible, impossible

Focus on the required attributes (e.g., color, shape, counts, spatial relations) of objects in the prompt.

Say "impossible" for a quarter only if it is certain that the prompt cannot be satisfied:
- the visible part already makes it clear that the prompt cannot be satisfied (e.g., too many objects are already drawn, or or an object has the wrong color or an incorrect shape), OR
- even if the lower three-quarters of that full image (not shown) were completed naturally, the final image would still not match the required attributes.

If there is any reasonable way the prompt could still be satisfied, say "possible".
"""


GPT_SELECT_SECOND_QUARTERS_PROMPT = """You are given a single image consisting of 2 contiguous halves (from top to bottom: half 1, half 2). 
Each half shows the top half of a different full image generated from the same text prompt. The bottom halves of those full images are not shown. 
Some objects may appear only partially (for example, the top half of an object is visible, and the bottom half would appear if the image is completed).

The text prompt is: "{}".

For each of the 2 halves, answer strictly with either "possible" or "impossible" (lowercase, no punctuation). 
Output must contain exactly 2 words, separated by a comma, in order: half 1, half 2. 
Example format: possible,impossible

Focus on the required attributes (e.g., color, shape, counts, spatial relations) of objects in the prompt.

Say "impossible" for a half only if it is certain that the prompt cannot be satisfied:
- the visible part already makes it clear that the prompt cannot be satisfied (e.g., too many objects are already drawn,or an object has the wrong color or an incorrect shape), OR
- even if the hidden bottom half of that full image were completed naturally, the final image would still not match the required attributes.

Otherwise, say "possible".
"""


REFINE_FIRST_PROMPT = """Rewrite the prompt "{}" considering the given partially generated images, so that it fully describes the final image layout with the correct total number of objects and accurately satisfies the required attributes (e.g., shape, color) in the original prompt, helping the model complete the remaining part.

RULES
- Keep the object type and total count exactly the same as in the original prompt.
- Do NOT add, remove, or change objects. Never alter the number.
- Do NOT introduce new attributes not present in the original prompt.
- Strictly preserve the original prompt at the beginning; only append a simple clause if it is directly useful (e.g., "<X> on the top and <Y> on the bottom").
- If the visible quarter already contains all required objects but only in incomplete form, leave the prompt unchanged; however, if the empty lower area could make the model add extras, append a minimal placement clause that locks the existing set (e.g., "centered in a single horizontal row").

FALLBACK
- If any rule would be violated, unfeasible, or uncertain, return the original prompt unchanged.

OUTPUT
- Output exactly one sentence: either the refined prompt or the unchanged original.
- Begin with the original text prompt; when refining, append the clause after a comma.

EXAMPLES
Original: "A photo of eight bears"; Visible: three bears on the top →
Output: "A photo of eight bears, three on the top and five on the bottom."

Original: "A photo of one chicken"; Visible: only the upper half of the same chicken →
Output: "A photo of one chicken" (unchanged; continuation only)

Original: "A green bench and a red car"; Visible: the upper part of the green bench →
Output: "A green bench and a red car, the green bench on the top and the red car on the bottom"

Original: "A photo of five plates"; Visible: the upper halves of the same five plates already spanning the frame →
Output: "A photo of five plates, centered in a single horizontal row."
"""


REFINE_SECOND_PROMPT = """Rewrite the prompt "{}" considering the given partially generated images, so that it fully describes the final image layout with the correct total number of objects and accurately satisfies the required attributes (e.g., shape, color) in the original prompt, helping the model complete the remaining part.

RULES
- Keep the object type and total count exactly the same as in the original prompt.
- Do NOT add, remove, or change objects. Never alter the number.
- Do NOT introduce new attributes not present in the original prompt.
- Strictly preserve the original prompt at the beginning; only append a simple clause if it is directly useful (e.g., "<X> on the top and <Y> on the bottom").
- If the visible half already contains all required objects but only in incomplete form, leave the prompt unchanged; however, if the empty lower area could make the model add extras, append a minimal placement clause that locks the existing set (e.g., "centered in a single horizontal row").

FALLBACK
- If any rule would be violated, unfeasible, or uncertain, return the original prompt unchanged.

OUTPUT
- Output exactly one sentence: either the refined prompt or the unchanged original.
- Begin with the original text prompt; when refining, append the clause after a comma.

EXAMPLES
Original: "A photo of eight bears"; Visible: three bears on the top →
Output: "A photo of eight bears, three on the top and five on the bottom."

Original: "A photo of one chicken"; Visible: only the upper half of the same chicken →
Output: "A photo of one chicken" (unchanged; continuation only)

Original: "A green bench and a red car"; Visible: the upper part of the green bench →
Output: "A green bench and a red car, the green bench on the top and the red car on the bottom"

Original: "A photo of five plates"; Visible: the upper halves of the same five plates already spanning the frame →
Output: "A photo of five plates, centered in a single horizontal row."
"""


@dataclass(frozen=True)
class PartialStageSpec:
    key: str
    rows_per_composite: int
    queries_per_candidate_group: int
    selection_prompt_template: str
    refinement_prompt_template: str


GRIDAR_PARTIAL_STAGES = (
    PartialStageSpec(
        key="after_first_quarter",
        rows_per_composite=4,
        queries_per_candidate_group=1,
        selection_prompt_template=GPT_SELECT_FIRST_QUARTERS_PROMPT,
        refinement_prompt_template=REFINE_FIRST_PROMPT,
    ),
    PartialStageSpec(
        key="after_first_half",
        rows_per_composite=2,
        queries_per_candidate_group=2,
        selection_prompt_template=GPT_SELECT_SECOND_QUARTERS_PROMPT,
        refinement_prompt_template=REFINE_SECOND_PROMPT,
    ),
)

def build_method_name(cfg: DictConfig, model_name: str) -> str:
    components = [
        cfg.method.name,
        model_name,
        f"{cfg.prm.prm_model}PRM",
        f"{cfg.orm.model_name}ORM",
        cfg.benchmark.name,
        f"cfg{cfg.cfg_scale}",
        f"group4{cfg.is_grouped_by_4}",
        f"N{cfg.N}",
    ]
    return "__".join(sanitize_component(component) for component in components)

@hydra.main(version_base=None, config_path="configs", config_name="evaluate_gridar")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    benchmark_adapter = validate_benchmark_config(cfg)
    if cfg.N % 4 != 0:
        raise ValueError("N must be divisible by 4.")

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
    if "gpt" in str(cfg.prm.prm_model).lower():
        prm_client = load_prm_client(
            cfg.prm.prm_model,
            api_key=cfg.prm.openai_api_key or os.environ.get("OPENAI_API_KEY"),
        )
    else:
        prm_client = load_prm_model(cfg.prm.prm_model)

    method_name = build_method_name(cfg, model_name)
    output_paths: OutputPaths = prepare_output_paths(paths.output_root, model_name, method_name)

    sample_records = load_sample_records(cfg, paths.dataset_path, benchmark_adapter)
    total_samples = len(sample_records)
    data_start_idx, data_end_idx = resolve_partition_range(total_samples, cfg.batch_size, cfg.partition_id, cfg.total_partitions)

    stats = {
        "n_possible_after_first_quarter": 0,
        "n_total_after_first_quarter": 0,
        "n_possible_after_first_half": 0,
        "n_total_after_first_half": 0,
        "n_all_impossible_after_first_quarter": 0,
        "n_all_impossible_after_first_half": 0,
    }

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
        candidate_names = build_candidate_names(batch_size, cfg.N)
        input_embeds, attention_mask = tokenize_generation_prompts(
            runtime,
            expanded_prompts,
            None,
            use_3way_cfg=False,
        )

        stage_start_time = time.time()

        quarter_cfg = cfg.method.after_first_quarter
        if quarter_cfg.use_3way_cfg and not quarter_cfg.refine:
            raise ValueError("after_first_quarter.use_3way_cfg requires refine=true.")
        quarter_spec = GRIDAR_PARTIAL_STAGES[0]
        quarter_request = build_t2i_request(cfg, input_embeds, attention_mask)
        quarter_raw_tokens = runtime.generator.generate_t2i_first_quarter(quarter_request)
        quarter_raw_images = runtime.generator.image_decode(quarter_request, quarter_raw_tokens).astype(np.uint8)
        quarter_selected_tokens = None
        quarter_selected_images = None
        quarter_verify_outputs = None
        quarter_num_possible = None
        quarter_num_all_impossible = None
        quarter_visible_images = None

        if quarter_cfg.select:
            (
                quarter_selected_tokens,
                quarter_selected_images,
                quarter_verify_outputs,
                quarter_num_possible,
                quarter_num_all_impossible,
            ) = run_stage_selection(
                cfg=cfg,
                stage_spec=quarter_spec,
                prm_client=prm_client,
                generation_prompts=expanded_prompts,
                raw_tokens=quarter_raw_tokens,
                raw_images=quarter_raw_images,
            )
            stats["n_possible_after_first_quarter"] += quarter_num_possible
            stats["n_total_after_first_quarter"] += batch_size * cfg.N
            stats["n_all_impossible_after_first_quarter"] += quarter_num_all_impossible

        quarter_tokens = quarter_selected_tokens if quarter_selected_tokens is not None else quarter_raw_tokens

        if quarter_cfg.refine:
            quarter_visible_images = build_visible_stage_images(
                quarter_selected_images if quarter_selected_images is not None else quarter_raw_images,
                cfg.model_params.img_size,
                quarter_spec.rows_per_composite,
            )
            quarter_refined_prompts = refine_stage_prompts(
                cfg=cfg,
                stage_spec=quarter_spec,
                prm_client=prm_client,
                generation_prompts=expanded_prompts,
                visible_images=quarter_visible_images,
            )
        else:
            quarter_refined_prompts = None
        quarter_prompts = quarter_refined_prompts or expanded_prompts

        input_embeds, attention_mask = tokenize_generation_prompts(
            runtime,
            expanded_prompts if quarter_cfg.use_3way_cfg else quarter_prompts,
            quarter_refined_prompts if quarter_cfg.use_3way_cfg else None,
            use_3way_cfg=quarter_cfg.use_3way_cfg,
        )

        half_cfg = cfg.method.after_first_half
        if half_cfg.use_3way_cfg and not half_cfg.refine:
            raise ValueError("after_first_half.use_3way_cfg requires refine=true.")
        half_spec = GRIDAR_PARTIAL_STAGES[1]
        half_request = build_t2i_request(cfg, input_embeds, attention_mask)
        half_raw_tokens = runtime.generator.generate_t2i_second_quarter(half_request, gen_tokens_q1=quarter_tokens)
        half_raw_images = runtime.generator.image_decode(half_request, half_raw_tokens).astype(np.uint8)
        half_selected_tokens = None
        half_selected_images = None
        half_verify_outputs = None
        half_num_possible = None
        half_num_all_impossible = None
        half_visible_images = None

        if half_cfg.select:
            (
                half_selected_tokens,
                half_selected_images,
                half_verify_outputs,
                half_num_possible,
                half_num_all_impossible,
            ) = run_stage_selection(
                cfg=cfg,
                stage_spec=half_spec,
                prm_client=prm_client,
                generation_prompts=expanded_prompts,
                raw_tokens=half_raw_tokens,
                raw_images=half_raw_images,
            )
            stats["n_possible_after_first_half"] += half_num_possible
            stats["n_total_after_first_half"] += batch_size * cfg.N
            stats["n_all_impossible_after_first_half"] += half_num_all_impossible

        half_tokens = half_selected_tokens if half_selected_tokens is not None else half_raw_tokens

        if half_cfg.refine:
            half_visible_images = build_visible_stage_images(
                half_selected_images if half_selected_images is not None else half_raw_images,
                cfg.model_params.img_size,
                half_spec.rows_per_composite,
            )
            half_refined_prompts = refine_stage_prompts(
                cfg=cfg,
                stage_spec=half_spec,
                prm_client=prm_client,
                generation_prompts=expanded_prompts,
                visible_images=half_visible_images,
            )
        else:
            half_refined_prompts = None
        half_prompts = half_refined_prompts or expanded_prompts

        input_embeds, attention_mask = tokenize_generation_prompts(
            runtime,
            expanded_prompts if half_cfg.use_3way_cfg else half_prompts,
            half_refined_prompts if half_cfg.use_3way_cfg else None,
            use_3way_cfg=half_cfg.use_3way_cfg,
        )

        final_request = build_t2i_request(cfg, input_embeds, attention_mask)
        final_tokens = runtime.generator.generate_t2i_second_half(final_request, gen_tokens_half=half_tokens)
        final_images = runtime.generator.image_decode(final_request, final_tokens).astype(np.uint8)

        token_prefix = output_paths.records_dir / f"batch{start_idx:04d}-{end_idx - 1}"
        quarter_select_path = Path(str(token_prefix) + "_quarter_select.txt")
        half_select_path = Path(str(token_prefix) + "_half_select.txt")
        quarter_refine_path = Path(str(token_prefix) + "_quarter_refine.txt")
        half_refine_path = Path(str(token_prefix) + "_half_refine.txt")
        if quarter_verify_outputs is not None:
            quarter_group_names = build_group_names(batch_size, cfg.N // quarter_spec.rows_per_composite)
            save_text_records(quarter_select_path, quarter_group_names, quarter_verify_outputs)
        if half_verify_outputs is not None:
            half_group_names = build_group_names(batch_size, cfg.N // half_spec.rows_per_composite)
            save_text_records(half_select_path, half_group_names, half_verify_outputs)
        if quarter_refined_prompts is not None:
            save_text_records(quarter_refine_path, candidate_names, quarter_refined_prompts)
        if half_refined_prompts is not None:
            save_text_records(half_refine_path, candidate_names, half_refined_prompts)
        torch.save(final_tokens.cpu(), str(token_prefix) + "_tokens.pt")

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

        if cfg.save_intermediate_outputs:
            quarter_group_names = build_group_names(batch_size, cfg.N // quarter_spec.rows_per_composite)
            half_group_names = build_group_names(batch_size, cfg.N // half_spec.rows_per_composite)
            save_intermediate_outputs(
                cfg=cfg,
                output_paths=output_paths,
                start_idx=start_idx,
                end_idx=end_idx,
                batch_prompts=batch_prompts,
                batch_names=batch_names,
                candidate_names=candidate_names,
                quarter_group_names=quarter_group_names,
                half_group_names=half_group_names,
                quarter_spec=quarter_spec,
                half_spec=half_spec,
                quarter_cfg=quarter_cfg,
                half_cfg=half_cfg,
                quarter_raw_images=quarter_raw_images,
                quarter_selected_images=quarter_selected_images,
                quarter_visible_images=quarter_visible_images,
                quarter_verify_outputs=quarter_verify_outputs,
                quarter_refined_prompts=quarter_refined_prompts,
                quarter_num_possible=quarter_num_possible,
                quarter_num_all_impossible=quarter_num_all_impossible,
                half_raw_images=half_raw_images,
                half_selected_images=half_selected_images,
                half_visible_images=half_visible_images,
                half_verify_outputs=half_verify_outputs,
                half_refined_prompts=half_refined_prompts,
                half_num_possible=half_num_possible,
                half_num_all_impossible=half_num_all_impossible,
                final_images=final_images,
                bon_scores=bon_scores,
                selected_indices=selected_indices,
                save_paths=save_paths,
            )

        elapsed = time.time() - stage_start_time
        print(f"Completed batch {start_idx}-{end_idx - 1} in {elapsed:.2f}s")

    save_config_snapshot(
        cfg,
        output_paths,
        stats,
        extras={
            "model_path": str(paths.model_path),
            "orm_model_path": str(paths.orm_model_path),
            "benchmark_root": str(paths.benchmark_root),
        },
    )


if __name__ == "__main__":
    main()
