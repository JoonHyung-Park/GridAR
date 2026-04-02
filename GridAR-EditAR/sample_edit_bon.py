import argparse
import os
import random
import time

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GRIDAR_ROOT = os.path.dirname(ROOT_DIR)
DEFAULT_PIE_BENCH_PATH = os.path.join(GRIDAR_ROOT, "benchmark", "PIE-Bench")
DEFAULT_DATASET_PATH = DEFAULT_PIE_BENCH_PATH
DEFAULT_EDITAR_CKPT_DIR = os.path.join(GRIDAR_ROOT, "checkpoints", "EditAR_ckpt")
DEFAULT_VQ_CKPT_PATH = os.path.join(DEFAULT_EDITAR_CKPT_DIR, "pretrained_models", "vq_ds16_t2i.pt")
DEFAULT_GPT_CKPT_PATH = os.path.join(DEFAULT_EDITAR_CKPT_DIR, "editar_release.pt")
DEFAULT_T5_PATH = os.path.join(DEFAULT_EDITAR_CKPT_DIR, "pretrained_models", "t5-ckpt")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_PIE_BENCH_PATH, "outputs")

from dataset_processor import PIE_Bench_Dataset
from editar.autoregressive.models.generate_edit import generate
from editar.autoregressive.models.gpt_edit import GPT_models
from editar.language.t5 import T5Embedder
from editar.tokenizer.tokenizer_image.vq_model import VQ_models

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@torch.inference_mode()
def tokenize_text(tokenizer, t5_model, prompts, device):
    text_tokens_and_mask = tokenizer(
        prompts,
        max_length=120,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_tokens_and_mask["input_ids"].to(device, non_blocking=True)
    input_ids_attn_mask = text_tokens_and_mask["attention_mask"].to(device, non_blocking=True)
    return t5_model.model(input_ids=input_ids, attention_mask=input_ids_attn_mask)["last_hidden_state"].detach()


def load_models(args, device):
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size**2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        model_mode=args.gpt_mode,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        distill_mode=args.distill_mode,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise ValueError("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)

    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    return vq_model, gpt_model, t5_model, latent_size


def main(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vq_model, gpt_model, t5_model, latent_size = load_models(args, device)

    save_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(os.path.join(save_path, "visualization"), exist_ok=True)

    dataset = PIE_Bench_Dataset(args, dataset_path=args.dataset_path, sample_index=args.sample_index)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    progress_bar = tqdm(total=len(dataset), desc="Processing", position=0)

    for idx, batch in enumerate(loader):
        image_id = dataset.mapping[idx]
        for seed in args.seed:
            tgt_path = os.path.join(save_path, "visualization", f"{image_id}_tgt_{seed}.png")
            if os.path.exists(tgt_path):
                print(f"skip {image_id} seed {seed}")
                continue

            random.seed(seed)
            torch.manual_seed(seed)

            input_img = batch["input_img"].to(device, non_blocking=True)
            input_mode = batch["mode"].to(device, non_blocking=True)
            prompts = list(batch["_edit_txt"])
            input_txt_embs = tokenize_text(t5_model.tokenizer, t5_model, prompts, device)

            with torch.no_grad():
                _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                input_img_indices = input_img_indices.reshape(input_img.shape[0], -1)

            qzshape = [input_img.shape[0], args.codebook_embed_dim, latent_size, latent_size]
            _ = time.time()
            index_sample = generate(
                gpt_model,
                input_txt_embs,
                input_img_indices,
                input_mode,
                latent_size**2,
                emb_masks=None,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                sample_logits=True,
            )

            samples = vq_model.decode_code(index_sample, qzshape)
            samples = torch.clamp(samples, -1, 1)

            src_image = Image.fromarray(np.uint8(batch["_input_img"][0].cpu().numpy()))
            tgt_image = (samples[0].cpu().permute(1, 2, 0) + 1) / 2 * 255.0
            tgt_image = Image.fromarray(np.uint8(tgt_image))

            src_image.save(os.path.join(save_path, "visualization", f"{image_id}_src.png"))
            tgt_image.save(tgt_path)
        progress_bar.update(1)

    progress_bar.close()
    print("Sampling is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill-mode", type=str, choices=["dinov2", "clip", "clipseg"], default=None)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=DEFAULT_VQ_CKPT_PATH)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=DEFAULT_GPT_CKPT_PATH)
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i", "edit"], default="edit")
    parser.add_argument("--gpt-mode", type=str, choices=["img_cls_emb", "joint_cls_emb"], default="joint_cls_emb")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=3)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--t5-path", type=str, default=DEFAULT_T5_PATH)
    parser.add_argument("--t5-model-type", type=str, default="flan-t5-xl")
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="BoN__EditAR__PIE__N4")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--sample-index", type=str, default=None)
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2, 3])
    main(parser.parse_args())
