import argparse
import csv
import json
import os

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GRIDAR_ROOT = os.path.dirname(ROOT_DIR)
DEFAULT_PIE_BENCH_PATH = os.path.join(GRIDAR_ROOT, "benchmark", "PIE-Bench")
DEFAULT_DATASET_PATH = DEFAULT_PIE_BENCH_PATH
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_PIE_BENCH_PATH, "outputs")
DEFAULT_QWEN_PATH = os.path.join(GRIDAR_ROOT, "checkpoints", "Qwen2.5-VL-7B-Instruct")


def resolve_mapping_file_path(dataset_path: str) -> str:
    candidate = os.path.join(dataset_path, "mapping_file.json")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Missing mapping file in {dataset_path}")


def find_candidate_paths(run_dir: str, sample_id: str, keys: list[int]):
    visualization_dir = os.path.join(run_dir, "visualization")
    candidate_paths = []
    for key in keys:
        path = os.path.join(visualization_dir, f"{sample_id}_tgt_{key}.png")
        if os.path.exists(path):
            candidate_paths.append((key, path))

    if candidate_paths:
        return candidate_paths
    raise FileNotFoundError(f"Could not resolve candidates for {sample_id} under {run_dir}")


def verify_image(editing_instruction, source_image_file, edited_image_file, model, processor):
    orm_prompt = (
        'The first image is the source image, and the second image is the edited image produced by '
        'applying the following instruction: "{}". Does the edited image correctly follow the '
        "instruction while preserving the rest of the source image? Please answer yes or no without explanation."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_image_file},
                {"type": "image", "image": edited_image_file},
                {"type": "text", "text": orm_prompt.format(editing_instruction)},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        cont = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True,
        )
    sequences = cont.sequences
    response = processor.tokenizer.convert_ids_to_tokens(sequences[0][input_len:])[0].lower()

    if response not in ["yes", "no"]:
        return response, 0.0

    yes_token_id = processor.tokenizer.convert_tokens_to_ids("yes")
    yes_token_id_2 = processor.tokenizer.convert_tokens_to_ids("Yes")
    scores = torch.cat([score.unsqueeze(1) for score in cont.scores], dim=1)
    scores = torch.nn.functional.softmax(scores, dim=-1)
    first_token_prob = scores[0, 0]
    yes_prob = first_token_prob[yes_token_id].item() + first_token_prob[yes_token_id_2].item()
    return response, yes_prob


def main(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        load_in_8bit=False,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    run_dir = os.path.join(args.output_dir, args.output_name)
    visualization_dir = os.path.join(run_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)

    with open(resolve_mapping_file_path(args.dataset_path), "r") as file:
        dataset = json.load(file)

    selection_csv_path = os.path.join(run_dir, "orm_selection.csv")
    with open(selection_csv_path, "w", newline="") as handle:
        csv.writer(handle).writerow(["id", "selected_key", "selected_score", "scores"])

    for sample_id, item in tqdm(dataset.items(), desc="ORM verify"):
        source_image_path = os.path.join(args.dataset_path, "annotation_images", item["image_path"])
        source_image = Image.open(source_image_path).convert("RGB")

        candidate_images = []
        candidate_scores = []
        candidate_paths = find_candidate_paths(run_dir, sample_id, args.key)
        for key, candidate_path in candidate_paths:
            candidate_image = Image.open(candidate_path).convert("RGB")
            _, score = verify_image(item["editing_instruction"], source_image, candidate_image, model, processor)
            candidate_images.append(candidate_image)
            candidate_scores.append(score)

        scores_tensor = torch.tensor(candidate_scores)
        selected_index = scores_tensor.argmax(dim=0).item()
        selected_key = candidate_paths[selected_index][0]
        selected_image = candidate_images[selected_index]

        save_path = os.path.join(visualization_dir, f"{sample_id}_tgt_best.png")
        selected_image.save(save_path)

        with open(selection_csv_path, "a", newline="") as handle:
            csv.writer(handle).writerow(
                [sample_id, selected_key, float(candidate_scores[selected_index]), json.dumps(candidate_scores)]
            )

    print(f"Saved selected images to {visualization_dir}")
    print(f"Saved ORM selection log to {selection_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--key", type=int, nargs="+", default=[0, 1, 2, 3])
    main(parser.parse_args())
