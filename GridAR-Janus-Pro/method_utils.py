import base64
import json
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from openai import OpenAI
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from generator import JanusGenerator, T2IRequest
from tokenize_embed import tokenize_text_janus, tokenize_text_janus_3_way


@dataclass(frozen=True)
class GenerationRuntime:
    model_name: str
    processor: Any
    model: Any
    generator: Any
    generation_device: torch.device


def load_generation_runtime(cfg, model_path: Path) -> GenerationRuntime:
    model_name = str(cfg.model_params.model_name)
    model_name_lower = model_name.lower()

    if "janus" in model_name_lower:
        from janus.models import VLChatProcessor

        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(str(model_path))
        vl_gpt = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True, device_map="auto")
        vl_gpt = vl_gpt.to(getattr(torch, cfg.model_params.model.dtype)).eval()
        generator = JanusGenerator(vl_gpt, cfg.model_params.max_batch_size)
        generation_device = next(vl_gpt.parameters()).device
        return GenerationRuntime(
            model_name=model_name,
            processor=vl_chat_processor,
            model=vl_gpt,
            generator=generator,
            generation_device=generation_device,
        )

    raise NotImplementedError(f"Unsupported backbone: {model_name}")


def tokenize_generation_prompts(
    runtime: GenerationRuntime,
    prompts: list[str],
    refined_prompts: Optional[list[str]],
    use_3way_cfg: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_name_lower = runtime.model_name.lower()

    if "janus" in model_name_lower:
        if use_3way_cfg:
            if refined_prompts is None:
                raise ValueError("Missing refined prompts.")
            return tokenize_text_janus_3_way(
                runtime.processor,
                runtime.model,
                prompts,
                refined_prompts,
                runtime.model_name,
                runtime.generation_device,
            )

        return tokenize_text_janus(
            runtime.processor,
            runtime.model,
            prompts,
            runtime.model_name,
            runtime.generation_device,
        )

    raise NotImplementedError(f"Unsupported backbone: {runtime.model_name}")


def build_candidate_names(batch_size: int, n: int) -> list[str]:
    return [f"p{prompt_idx:02d}_c{cand_idx:02d}.png" for prompt_idx in range(batch_size) for cand_idx in range(n)]


def build_group_names(batch_size: int, groups_per_prompt: int) -> list[str]:
    return [f"p{prompt_idx:02d}_g{group_idx:02d}.png" for prompt_idx in range(batch_size) for group_idx in range(groups_per_prompt)]


def save_text_records(path: Path, names: list[str], values: Optional[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if values is None:
        return

    with path.open("w", encoding="utf-8") as fp:
        for name, value in zip(names, values):
            fp.write(f"[{name}]\n{value}\n\n")


def save_image_records(directory: Path, names: list[str], images: Optional[np.ndarray]) -> None:
    if images is None:
        return

    directory.mkdir(parents=True, exist_ok=True)
    for name, image in zip(names, images):
        Image.fromarray(image.astype("uint8")).save(directory / name)


def encode_pil_image(pil_image: Image.Image) -> str:
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_prm_client(model_name: str, api_key: Optional[str] = None) -> Optional[Any]:
    model_name_lower = model_name.lower()
    if "gpt" in model_name_lower:
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("Set OPENAI_API_KEY.")
        return OpenAI(api_key=resolved_api_key)
    return None


def load_prm_model(model_name: str) -> Any:
    raise NotImplementedError(f"Unsupported PRM: {model_name}")


def query_prm(model_name: str, instruction: str, image: np.ndarray, prm_client: Optional[Any] = None) -> str:
    model_name_lower = model_name.lower()
    if "gpt" in model_name_lower:
        client = prm_client or load_prm_client(model_name)
        base64_image = encode_pil_image(Image.fromarray(image.astype("uint8")))
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                    ],
                }
            ],
        )
        return response.output_text.strip()
    else:
        raise NotImplementedError(f"Unsupported PRM: {model_name}")


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    float_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=float_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def score_candidate_with_orm(prompt: str, image_file: Image.Image, orm_model, orm_processor) -> tuple[bool, float]:
    orm_prompt = (
        "This image is generated by a prompt: {}. "
        "Does this image accurately represent the prompt? Please answer yes or no without explanation."
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_file},
                {"type": "text", "text": orm_prompt.format(prompt)},
            ],
        },
    ]

    inputs = orm_processor.apply_chat_template(
        [messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    orm_device = next(orm_model.parameters()).device
    inputs = move_batch_to_device(inputs, orm_device)
    input_len = inputs["input_ids"].shape[-1]

    yes_token_id = orm_processor.tokenizer.convert_tokens_to_ids("yes")
    yes_token_id_cap = orm_processor.tokenizer.convert_tokens_to_ids("Yes")

    with torch.no_grad():
        result = orm_model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = result.sequences
    answer = orm_processor.tokenizer.convert_ids_to_tokens(sequences[0][input_len:])[0].lower()
    if answer not in {"yes", "no"}:
        return False, 50.0

    scores = torch.cat([score.unsqueeze(1) for score in result.scores], dim=1)
    scores = torch.nn.functional.softmax(scores, dim=-1)
    first_token_prob = scores[0, 0]
    yes_prob = first_token_prob[yes_token_id].item() + first_token_prob[yes_token_id_cap].item()
    return answer == "yes", yes_prob


def run_orm_ranking(
    batch_prompts: list[str],
    images_bon: np.ndarray,
    orm_model,
    orm_processor,
) -> torch.Tensor:
    batch_size, bon = images_bon.shape[:2]
    orm_device = next(orm_model.parameters()).device
    scores = torch.zeros(batch_size, bon, device=orm_device)

    for batch_idx in range(batch_size):
        for cand_idx in range(bon):
            candidate_array = images_bon[batch_idx, cand_idx]
            candidate_image = Image.fromarray(candidate_array.astype("uint8"))
            _, score = score_candidate_with_orm(batch_prompts[batch_idx], candidate_image, orm_model, orm_processor)
            scores[batch_idx, cand_idx] = score

    return scores


def build_t2i_request(cfg, input_embeds: torch.Tensor, attention_mask: torch.Tensor) -> T2IRequest:
    return T2IRequest(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        image_token_num_per_image=cfg.model_params.image_token_num_per_image,
        img_size=cfg.model_params.img_size,
        patch_size=cfg.model_params.patch_size,
        temperature=getattr(cfg.model_params, "temperature", 1.0),
        top_k=getattr(cfg.model_params, "top_k", None),
        top_p=getattr(cfg.model_params, "top_p", None),
        pad=cfg.pad,
        cfg=cfg,
    )


def load_orm_model(orm_model_path: Path):
    orm_path_lower = str(orm_model_path).lower()

    if "qwen" in orm_path_lower:
        from transformers import Qwen2_5_VLForConditionalGeneration

        orm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(orm_model_path),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        orm_processor = AutoProcessor.from_pretrained(str(orm_model_path))
        return orm_model, orm_processor

    raise NotImplementedError(f"Unsupported ORM: {orm_model_path}")


def add_horizontal_lines(images: np.ndarray, num_rows: int) -> np.ndarray:
    if images.ndim == 3:
        images = images[np.newaxis, ...]

    output = images.copy()
    height = output.shape[1]
    line_positions = [(row * height) // num_rows for row in range(1, num_rows)]

    for image in output:
        for line_pos in line_positions:
            start = max(0, line_pos - 1)
            end = min(height, line_pos + 2)
            image[start:end, :, :] = 0
    return output


def compose_stage_grid(images: np.ndarray, visible_height: int, rows_per_composite: int) -> np.ndarray:
    if images.shape[0] % rows_per_composite != 0:
        raise ValueError("The number of images must be divisible by rows_per_composite")

    num_groups = images.shape[0] // rows_per_composite
    composed = np.zeros((num_groups, images.shape[1], images.shape[2], images.shape[3]), dtype=np.uint8)

    for group_idx in range(num_groups):
        group = images[group_idx * rows_per_composite:(group_idx + 1) * rows_per_composite]
        for item_idx in range(rows_per_composite):
            top = visible_height * item_idx
            bottom = visible_height * (item_idx + 1)
            composed[group_idx, top:bottom, :, :] = group[item_idx, :visible_height, :, :]

    return add_horizontal_lines(composed, num_rows=rows_per_composite)


def build_visible_stage_images(raw_images: np.ndarray, image_size: int, rows_per_composite: int) -> np.ndarray:
    visible_height = image_size // rows_per_composite
    visible_images = raw_images.copy()
    visible_images[:, visible_height:, :, :] = 0
    return add_horizontal_lines(visible_images, num_rows=rows_per_composite)


def build_partial_stage_visuals(raw_images: np.ndarray, image_size: int, rows_per_composite: int) -> tuple[np.ndarray, np.ndarray]:
    visible_height = image_size // rows_per_composite
    visible_images = build_visible_stage_images(raw_images, image_size, rows_per_composite)
    aggregate_images = compose_stage_grid(raw_images, visible_height, rows_per_composite)
    return visible_images, aggregate_images


def parse_possible_flags(raw_texts: list[str], expected: int) -> list[bool]:
    flags: list[bool] = []
    for raw_text in raw_texts:
        cleaned = raw_text.lower()
        for token in cleaned.split(","):
            flags.append(token.strip() == "possible")

    if len(flags) < expected:
        flags.extend([False] * (expected - len(flags)))
    return flags[:expected]


def normalize_refined_prompt(text: str, original_prompt: str) -> str:
    candidate = " ".join(text.strip().split())
    if not candidate:
        print("[normalize_refined_prompt] Empty refined prompt. Falling back to the original prompt.", file=sys.stderr)
        return original_prompt

    if '"' in candidate:
        first_quote = candidate.find('"')
        last_quote = candidate.rfind('"')
        if first_quote < last_quote:
            candidate = candidate[first_quote + 1:last_quote].strip()
        else:
            candidate = candidate.replace('"', "").strip()

    if not candidate:
        print("[normalize_refined_prompt] Invalid refined prompt after quote cleanup. Falling back to the original prompt.", file=sys.stderr)
        return original_prompt

    return candidate


def run_stage_selection(
    cfg,
    stage_spec,
    prm_client,
    generation_prompts: list[str],
    raw_tokens: torch.Tensor,
    raw_images: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, list[str], int, int]:
    _, aggregate_images = build_partial_stage_visuals(
        raw_images,
        cfg.model_params.img_size,
        stage_spec.rows_per_composite,
    )

    aggregate_prompts = generation_prompts[::stage_spec.rows_per_composite]
    verify_outputs = [
        query_prm(
            cfg.prm.prm_model,
            stage_spec.selection_prompt_template.format(prompt),
            aggregate_images[idx],
            prm_client=prm_client,
        ).lower()
        for idx, prompt in enumerate(aggregate_prompts)
    ]

    selected_tokens, selected_images, num_possible, num_all_impossible = propagate_selection(
        gen_tokens=raw_tokens,
        verify_outputs=verify_outputs,
        batch_size=len(generation_prompts) // cfg.N,
        n=cfg.N,
        grouped_by_4=cfg.is_grouped_by_4,
        images=raw_images,
        queries_per_candidate_group=stage_spec.queries_per_candidate_group,
    )

    return selected_tokens, selected_images, verify_outputs, num_possible, num_all_impossible


def refine_stage_prompts(
    cfg,
    stage_spec,
    prm_client,
    generation_prompts: list[str],
    visible_images: np.ndarray,
) -> list[str]:
    refined_prompts = []
    for idx, prompt in enumerate(generation_prompts):
        response = query_prm(
            cfg.prm.prm_model,
            stage_spec.refinement_prompt_template.format(prompt),
            visible_images[idx],
            prm_client=prm_client,
        )
        refined_prompts.append(normalize_refined_prompt(response, prompt))
    return refined_prompts


def propagate_selection(
    gen_tokens: torch.Tensor,
    verify_outputs: list[str],
    batch_size: int,
    n: int,
    grouped_by_4: bool,
    images: np.ndarray,
    queries_per_candidate_group: int,
) -> tuple[torch.Tensor, np.ndarray, int, int]:
    if n % 4 != 0:
        raise ValueError(f"N ({n}) must be divisible by 4")

    candidate_groups_per_prompt = n // 4
    expected_outputs = batch_size * candidate_groups_per_prompt * queries_per_candidate_group
    if len(verify_outputs) != expected_outputs:
        raise ValueError(f"Expected {expected_outputs} verifier outputs, got {len(verify_outputs)}")

    gen_tokens = gen_tokens.clone()
    images = images.copy()
    num_possible = 0
    num_all_impossible = 0

    for prompt_idx in range(batch_size):
        group_labels: list[list[bool]] = []
        possible_indices: list[int] = []
        impossible_indices: list[int] = []

        for group_idx in range(candidate_groups_per_prompt):
            if queries_per_candidate_group == 1:
                raw_group = [verify_outputs[prompt_idx * candidate_groups_per_prompt + group_idx]]
            else:
                base_idx = (prompt_idx * candidate_groups_per_prompt + group_idx) * queries_per_candidate_group
                raw_group = verify_outputs[base_idx:base_idx + queries_per_candidate_group]

            flags = parse_possible_flags(raw_group, expected=4)
            group_labels.append(flags)

            base = prompt_idx * n + group_idx * 4
            for offset, is_possible in enumerate(flags):
                index = base + offset
                if is_possible:
                    possible_indices.append(index)
                else:
                    impossible_indices.append(index)

        if grouped_by_4:
            for group_idx, flags in enumerate(group_labels):
                base = prompt_idx * n + group_idx * 4
                possible_local = [offset for offset, is_possible in enumerate(flags) if is_possible]
                impossible_local = [offset for offset, is_possible in enumerate(flags) if not is_possible]
                if not possible_local:
                    num_all_impossible += 1
                    continue
                for replace_idx, target_offset in enumerate(impossible_local):
                    source_offset = possible_local[replace_idx % len(possible_local)]
                    src = base + source_offset
                    dst = base + target_offset
                    gen_tokens[dst] = gen_tokens[src]
                    images[dst] = images[src]
        else:
            if not possible_indices:
                num_all_impossible += 1
                continue
            for replace_idx, dst in enumerate(impossible_indices):
                src = possible_indices[replace_idx % len(possible_indices)]
                gen_tokens[dst] = gen_tokens[src]
                images[dst] = images[src]

        num_possible += len(possible_indices)

    return gen_tokens, images, num_possible, num_all_impossible


def save_intermediate_outputs(
    cfg,
    output_paths,
    start_idx: int,
    end_idx: int,
    batch_prompts: list[str],
    batch_names: list[str],
    candidate_names: list[str],
    quarter_group_names: list[str],
    half_group_names: list[str],
    quarter_spec,
    half_spec,
    quarter_cfg,
    half_cfg,
    quarter_raw_images: np.ndarray,
    quarter_selected_images: Optional[np.ndarray],
    quarter_visible_images: Optional[np.ndarray],
    quarter_verify_outputs: Optional[list[str]],
    quarter_refined_prompts: Optional[list[str]],
    quarter_num_possible: Optional[int],
    quarter_num_all_impossible: Optional[int],
    half_raw_images: np.ndarray,
    half_selected_images: Optional[np.ndarray],
    half_visible_images: Optional[np.ndarray],
    half_verify_outputs: Optional[list[str]],
    half_refined_prompts: Optional[list[str]],
    half_num_possible: Optional[int],
    half_num_all_impossible: Optional[int],
    final_images: np.ndarray,
    bon_scores: torch.Tensor,
    selected_indices: torch.Tensor,
    save_paths: list[Path],
) -> None:
    batch_dir = output_paths.sample_dir.parent / "intermediate_outputs" / f"batch{start_idx:04d}-{end_idx - 1}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    _, quarter_aggregate_images = build_partial_stage_visuals(
        quarter_raw_images,
        cfg.model_params.img_size,
        quarter_spec.rows_per_composite,
    )
    _, half_aggregate_images = build_partial_stage_visuals(
        half_raw_images,
        cfg.model_params.img_size,
        half_spec.rows_per_composite,
    )

    save_image_records(batch_dir / "quarter_raw", candidate_names, quarter_raw_images)
    save_image_records(batch_dir / "quarter_selected", candidate_names, quarter_selected_images)
    save_image_records(batch_dir / "quarter_visible", candidate_names, quarter_visible_images)
    save_image_records(batch_dir / "quarter_aggregate", quarter_group_names, quarter_aggregate_images)
    save_image_records(batch_dir / "half_raw", candidate_names, half_raw_images)
    save_image_records(batch_dir / "half_selected", candidate_names, half_selected_images)
    save_image_records(batch_dir / "half_visible", candidate_names, half_visible_images)
    save_image_records(batch_dir / "half_aggregate", half_group_names, half_aggregate_images)
    save_image_records(batch_dir / "final_candidates", candidate_names, final_images)

    if quarter_verify_outputs is not None:
        save_text_records(batch_dir / "quarter_select.txt", quarter_group_names, quarter_verify_outputs)
    if half_verify_outputs is not None:
        save_text_records(batch_dir / "half_select.txt", half_group_names, half_verify_outputs)
    if quarter_refined_prompts is not None:
        save_text_records(batch_dir / "quarter_refine.txt", candidate_names, quarter_refined_prompts)
    if half_refined_prompts is not None:
        save_text_records(batch_dir / "half_refine.txt", candidate_names, half_refined_prompts)

    summary = {
        "batch_range": [start_idx, end_idx - 1],
        "batch_prompts": batch_prompts,
        "batch_names": batch_names,
        "candidate_names": candidate_names,
        "quarter": {
            "select": bool(quarter_cfg.select),
            "refine": bool(quarter_cfg.refine),
            "use_3way_cfg": bool(quarter_cfg.use_3way_cfg),
            "num_possible": quarter_num_possible,
            "num_all_impossible": quarter_num_all_impossible,
            "verify_outputs": quarter_verify_outputs,
            "refined_prompts": quarter_refined_prompts,
        },
        "half": {
            "select": bool(half_cfg.select),
            "refine": bool(half_cfg.refine),
            "use_3way_cfg": bool(half_cfg.use_3way_cfg),
            "num_possible": half_num_possible,
            "num_all_impossible": half_num_all_impossible,
            "verify_outputs": half_verify_outputs,
            "refined_prompts": half_refined_prompts,
        },
        "orm_scores": bon_scores.tolist(),
        "selected_indices": selected_indices.tolist(),
        "save_paths": [str(path) for path in save_paths],
    }
    with (batch_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
