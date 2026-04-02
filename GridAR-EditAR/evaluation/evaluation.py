import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image
import torch

from metrics_calculator import MetricsCalculator


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROOT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from dataset_processor import resolve_mapping_file_path


GRIDAR_ROOT = os.path.dirname(PROJECT_DIR)
DEFAULT_DATASET_PATH = os.path.join(GRIDAR_ROOT, "benchmark", "PIE-Bench")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATASET_PATH, "outputs")


def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask, src_prompt, tgt_prompt):
    if metric == "psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric == "lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric == "mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric == "ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric == "psnr_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_psnr(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "lpips_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_lpips(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "mse_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_mse(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "ssim_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_ssim(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "structure_distance_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "psnr_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "lpips_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "mse_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "ssim_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "structure_distance_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)
    raise ValueError(f"Unknown metric: {metric}")


def mask_decode(encoded_mask, image_shape=(512, 512)):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1
    return mask_array


def resolve_target_path(target_dir, sample_id, key):
    if key == "best":
        best_path = os.path.join(target_dir, "visualization", f"{sample_id}_tgt_best.png")
        if os.path.exists(best_path):
            return "best", best_path
        raise FileNotFoundError(
            f"Could not find verifier-selected target for {sample_id} under {target_dir}. "
            "Run orm_verifier.py first."
        )

    bon_path = os.path.join(target_dir, "visualization", f"{sample_id}_tgt_{key}.png")
    if os.path.exists(bon_path):
        return key, bon_path
    raise FileNotFoundError(f"Could not resolve target for {sample_id} key {key} under {target_dir}")


def main(args):
    metrics = [
        "structure_distance",
        "psnr_unedit_part",
        "lpips_unedit_part",
        "mse_unedit_part",
        "ssim_unedit_part",
        "clip_similarity_target_image",
        "clip_similarity_target_image_edit_part",
    ]
    metrics_calculator = MetricsCalculator("cuda" if torch.cuda.is_available() else "cpu")

    target_dir = os.path.join(args.output_dir, args.test_name)
    suffix = args.key
    save_name = f"eval_{args.test_name}_{suffix}"
    if args.miniset:
        save_name += "_mini"
    else:
        save_name += "_full"
    save_path = os.path.join(args.output_dir, f"{save_name}.csv")
    if os.path.exists(save_path):
        os.remove(save_path)

    with open(resolve_mapping_file_path(args.dataset_path), "r") as file:
        dataset = json.load(file)
    if args.miniset:
        dataset = {sample_id: item for sample_id, item in dataset.items() if sample_id.startswith("0")}

    header = ["id", "target_key"] + metrics
    with open(save_path, "w", newline="") as handle:
        csv.writer(handle).writerow(header)

    metric_values = {metric: [] for metric in metrics}
    for sample_id, item in dataset.items():
        src_path = item["image_path"]
        src_image = Image.open(os.path.join(args.dataset_path, "annotation_images", src_path)).convert("RGB")
        src_prompt = item["original_prompt"]
        tgt_prompt = item["editing_prompt"]
        mask = mask_decode(item["mask"])[:, :, np.newaxis].repeat([3], axis=2)

        target_key, target_path = resolve_target_path(target_dir, sample_id, args.key)
        tgt_image = Image.open(target_path).convert("RGB")

        if args.save_best_dir and args.save_best_name:
            os.makedirs(args.save_best_dir, exist_ok=True)
            src_image.save(os.path.join(args.save_best_dir, f"{sample_id}_src.png"))
            tgt_image.save(os.path.join(args.save_best_dir, f"{sample_id}_{args.save_best_name}.png"))

        evaluation_result = [sample_id, target_key]
        for metric in metrics:
            cal_metric = calculate_metric(
                metrics_calculator,
                metric,
                src_image=src_image,
                tgt_image=tgt_image,
                src_mask=mask,
                tgt_mask=mask,
                src_prompt=src_prompt,
                tgt_prompt=tgt_prompt,
            )
            if cal_metric == "nan":
                evaluation_result.append(cal_metric)
                continue
            if "mse" in metric:
                cal_metric *= 10000
            if "lpips" in metric:
                cal_metric *= 1000
            if "ssim" in metric:
                cal_metric *= 100
            if "structure_distance" in metric:
                cal_metric *= 1000
            evaluation_result.append(cal_metric)
            metric_values[metric].append(cal_metric)

        with open(save_path, "a+", newline="") as handle:
            csv.writer(handle).writerow(evaluation_result)

    with open(save_path, "a+", newline="") as handle:
        writer = csv.writer(handle)
        mean_row = ["mean", suffix]
        for metric in metrics:
            values = metric_values[metric]
            mean_row.append(sum(values) / len(values) if values else "nan")
        writer.writerow(mean_row)

    print(f"Saved evaluation results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory containing run outputs")
    parser.add_argument("--test_name", type=str, required=True, help="Name of the run folder")
    parser.add_argument("--key", type=str, default="best", help='Candidate key to evaluate, or "best"')
    parser.add_argument("--miniset", action="store_true")
    parser.add_argument("--save_best_dir", type=str, default="")
    parser.add_argument("--save_best_name", type=str, default="")
    main(parser.parse_args())
