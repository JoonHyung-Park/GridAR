import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GRIDAR_ROOT = os.path.dirname(ROOT_DIR)
DEFAULT_PIE_BENCH_PATH = os.path.join(GRIDAR_ROOT, "benchmark", "PIE-Bench")
DEFAULT_DATASET_PATH = DEFAULT_PIE_BENCH_PATH
DEFAULT_EDITAR_CKPT_DIR = os.path.join(GRIDAR_ROOT, "checkpoints", "EditAR_ckpt")
DEFAULT_VQ_CKPT_PATH = os.path.join(DEFAULT_EDITAR_CKPT_DIR, "pretrained_models", "vq_ds16_t2i.pt")
DEFAULT_GPT_CKPT_PATH = os.path.join(DEFAULT_EDITAR_CKPT_DIR, "editar_release.pt")
DEFAULT_T5_PATH = os.path.join(DEFAULT_EDITAR_CKPT_DIR, "pretrained_models", "t5-ckpt")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_PIE_BENCH_PATH, "outputs")

import argparse
import random
from editar.tokenizer.tokenizer_image.vq_model import VQ_models
from editar.language.t5 import T5Embedder
from editar.autoregressive.models.gpt_edit import GPT_models
from generate_edit_gridar import generate_edit_quarter, generate_edit_second_quarter, grid_generate_edit_third_stage
from torch.utils.data import DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm


import numpy as np
from PIL import Image
import json
from openai import OpenAI
from io import BytesIO
import base64
from typing import List

from dataset_processor import PIE_Bench_Dataset
from gpt_prompts import gpt_select_first_row_prompt, gpt_select_second_row_prompt, refine_prompt

client = None


def build_openai_client(api_key: str | None = None) -> OpenAI:
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("Set OPENAI_API_KEY.")
    return OpenAI(api_key=resolved_api_key)

def add_horizontal_lines(images: np.ndarray, num_rows: int = 4) -> np.ndarray:

    if images.ndim == 3:        # (H, W, C)
        images = images[np.newaxis, ...]  # (1, H, W, C)

    h = images.shape[1]
    line_positions = [(i * h) // num_rows for i in range(1, num_rows)]

    for img in images:
        for y in line_positions:
            img[y-1:y + 2, :, :] = 0  

    return images

def encode_pil_image(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  
    return base64.b64encode(buffered.getvalue()).decode("utf-8")    


def save_text_record(record_dir: str, filename: str, values: list[str]) -> None:
    with open(os.path.join(record_dir, filename), "w", encoding="utf-8") as handle:
        handle.write("\n".join(values))

def propagate_selection_flexible(
    gen_tokens: torch.Tensor,
    verify_texts: List[str],
    batch_size: int,
    N: int,
    grouped_by_4: bool,
    images: np.ndarray,
    aggregate_images: np.ndarray,
    record_dir: str,
    sample_ids: str,
    prompts_per_group: int = 4,   # <-- 추가: 기본값 4
):
    assert N % 4 == 0, f"N({N}) must be a multiple of 4."
    groups_per_prompt = N // 4
    if prompts_per_group == 4:
        expected = batch_size * groups_per_prompt
    elif prompts_per_group == 2:
        expected = batch_size * groups_per_prompt * 2
    else:
        raise ValueError("prompts_per_group must be 2 or 4")
    
    assert len(verify_texts) == expected, \
        f"verify_texts len mismatch: expect {expected}, got {len(verify_texts)}"
        
    gen_tokens = gen_tokens.clone()
    images = images.copy()
    def parse_flags(txts: List[str]) -> List[bool]:
        flags_all = []
        save_txts_IF, save_txts_SP = [], []
        for txt in txts:
            instruction_following = json.loads(txt)["instruction_following"].split(",")
            source_preservation = json.loads(txt)["source_preservation"].split(",")
            save_output_text_IF = ''.join([t.strip()[0] for t in instruction_following])
            save_output_text_SP = ''.join([t.strip()[0] for t in source_preservation])
            flags = [instruction_following[i].strip() == "pass" and source_preservation[i].strip() == "pass" \
                     for i in range(len(instruction_following))]
            flags_all.extend(flags)
            save_txts_IF.append(save_output_text_IF)
            save_txts_SP.append(save_output_text_SP)
        return flags_all, save_txts_IF, save_txts_SP

    group_labels = []
    possible_abs, impossible_abs = [], []

        
    for g in range(groups_per_prompt):
        if prompts_per_group == 4:
            txts = [verify_texts[g]]
            flags, save_txts_IF, save_txts_SP = parse_flags(txts)
            group_labels.append(flags)
            save_output_text = save_txts_IF[0] + "_" + save_txts_SP[0]
            Image.fromarray(aggregate_images[g]).save(
                os.path.join(record_dir, f"{sample_ids}-quarters_{g}-{save_output_text}.png")
            )
        else:  # prompts_per_group == 2
            base_idx = g * 2
            txts = [verify_texts[base_idx], verify_texts[base_idx + 1]]
            flags, save_txts_IF, save_txts_SP = parse_flags(txts)
            group_labels.append(flags)
            for i in range(len(txts)):
                save_output_text = save_txts_IF[i] + "_" + save_txts_SP[i]
                Image.fromarray(aggregate_images[base_idx + i]).save(
                    os.path.join(record_dir, f"{sample_ids}-halves_{g}_{i}-{save_output_text}.png")
                )

        base = g * 4
        for j, ok in enumerate(flags):
            idx = base + j
            (possible_abs if ok else impossible_abs).append(idx)

    if grouped_by_4:
        # propagate within each group of 4
        for g, flags in enumerate(group_labels):
            base = g * 4
            poss_local = [j for j, ok in enumerate(flags) if ok]
            imp_local  = [j for j, ok in enumerate(flags) if not ok]
            if len(poss_local) == 0:
                continue
            for k, j in enumerate(imp_local):
                src_local = poss_local[k % len(poss_local)]
                src, dst = base + src_local, base + j
                gen_tokens[dst] = gen_tokens[src]
                images[dst] = images[src]
    else:
        # propagate across all groups for the prompt
        if len(possible_abs) > 0:
            for k, dst in enumerate(impossible_abs):
                src = possible_abs[k % len(possible_abs)]
                gen_tokens[dst] = gen_tokens[src]
                images[dst] = images[src]
    
    return gen_tokens, images
    
@torch.inference_mode()
def tokenize_text (llm_tokenizer, t5_model, prompts, device):
    text_tokens_and_mask = llm_tokenizer(
        prompts,
        max_length=120,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    txt_token = text_tokens_and_mask['input_ids']
    txt_attn_mask = text_tokens_and_mask['attention_mask']

    input_ids = txt_token[0].unsqueeze(0).to(device, non_blocking=True)
    input_ids_attn_mask = txt_attn_mask[0].unsqueeze(0).to(device, non_blocking=True)

    input_txt_embs = t5_model.model(
        input_ids=input_ids,
        attention_mask=input_ids_attn_mask,
    )['last_hidden_state'].detach()
    
    return input_txt_embs


def main(args):
    global client
    random.seed(args.seed)
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = build_openai_client(args.openai_api_key)

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
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
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    
    save_path = os.path.join(args.output_dir, args.output_name)
    visualization_dir = os.path.join(save_path, "visualization")
    records_dir = os.path.join(save_path, "records")
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(records_dir, exist_ok=True)
    # dataset = Image_Folder_Dataset(args, dataset_path=save_path, llm_tokenizer=t5_model.tokenizer, mode='val')
    dataset = PIE_Bench_Dataset(args, 
                                dataset_path=args.dataset_path,
                                sample_index=args.sample_index)
    
    batch_size = args.batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    N = args.N
    
    progress_bar = tqdm(total=len(dataset), desc="Processing", position=0)
    for id, batch in enumerate(loader):
        sample_ids = dataset.mapping[id]
        if os.path.exists(os.path.join(visualization_dir, f"{sample_ids}_tgt_0.png")):
            print(f"skip {sample_ids}")
            progress_bar.update(1)
            continue
        input_img = batch['input_img'].to(device, non_blocking=True)
        input_mode = batch['mode'].to(device, non_blocking=True)
        prompts = batch['_edit_txt']
        img_size = batch['_input_img'].shape[1]
        quarter_img_size = img_size // 4

        input_txt_embs = tokenize_text(
            t5_model.tokenizer, t5_model, prompts, device
        )
        # process image ids to embeddings
        with torch.no_grad():
            _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
            input_img_indices = input_img_indices.reshape(input_img.shape[0], -1)

        # =================================================
        # ################## First Stage ##################
        # =================================================
        
        B = input_img.shape[0]
        assert B == 1, "currently only support batch size 1 for inference"
        
        input_txt_embs = input_txt_embs.repeat(N, 1, 1)
        input_img_indices = input_img_indices.repeat(N, 1)
        input_mode = input_mode.repeat(N, 1)

        with torch.inference_mode():
            gen_tokens = generate_edit_quarter(
                gpt_model, input_txt_embs, input_img_indices, input_mode, latent_size ** 2, 
                emb_masks=None, 
                cfg_scale=args.cfg_scale,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True, 
                )
            
        qz_shape = [1, 8, latent_size, latent_size]
        
        q1_images = np.zeros((B * N, img_size, img_size, 3), dtype=np.uint8)
        
        for i, image_token in enumerate(gen_tokens):
            img = vq_model.decode_code([image_token], qz_shape)[0]
            img = img.to(torch.float32).cpu().numpy().transpose(1, 2, 0)
                    
            norm_img = (img + 1) / 2 * 255
            norm_img = norm_img.clip(0, 255)
            
            q1_images[i] = norm_img
        
        q1_images = add_horizontal_lines(q1_images, num_rows=4)
        
        num_aggregate_q1_images = B * N // 4
        aggregate_q1_images = np.zeros((num_aggregate_q1_images, img_size, img_size, 3), dtype=np.uint8) 
            
        for prompt_idx in range(num_aggregate_q1_images):
            quarters = q1_images[4*prompt_idx:4*(prompt_idx+1)]  
            for i in range(4):
                aggregate_q1_images[prompt_idx,quarter_img_size*i:quarter_img_size*(i+1),:,:] = quarters[i, :quarter_img_size, :, :]
                
        aggregate_q1_images = add_horizontal_lines(aggregate_q1_images, num_rows=4)
        
        outputs = []
        base64_input_image = encode_pil_image(Image.fromarray(np.array(batch['_input_img'])[0]))
        for img_idx in range(num_aggregate_q1_images):
            base64_output_image = encode_pil_image(Image.fromarray(aggregate_q1_images[img_idx]))
            response = client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": gpt_select_first_row_prompt.format(prompts[0])},
                            # (A) Source image 
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_input_image}",
                            },
                            # (B) Composite (4 quarters) 
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_output_image}",
                            },
                        ],
                    }
                ],
            )
            outputs.append(response.output_text)
        outputs = [o.lower() for o in outputs]
        save_text_record(records_dir, f"{sample_ids}-quarter_select.txt", outputs)
        
        gen_tokens, _ = propagate_selection_flexible(
            gen_tokens,
            verify_texts=outputs,
            batch_size=B,
            N=N,
            grouped_by_4=args.grouped_by_4,
            images=q1_images,
            aggregate_images=aggregate_q1_images,
            record_dir=records_dir,
            sample_ids=sample_ids,
            prompts_per_group=4,
        )

        with torch.inference_mode():
            gen_tokens = generate_edit_second_quarter(
                gpt_model, input_txt_embs, input_img_indices, input_mode, 
                gen_tokens=gen_tokens,
                image_token_num_per_image=latent_size ** 2, 
                emb_masks=None, 
                cfg_scale=args.cfg_scale,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True, 
                )
            
        qz_shape = [1, 8, latent_size, latent_size]
        
        q2_images = np.zeros((B * N, img_size, img_size, 3), dtype=np.uint8)
        
        for i, image_token in enumerate(gen_tokens):
            img = vq_model.decode_code([image_token], qz_shape)[0]
            img = img.to(torch.float32).cpu().numpy().transpose(1, 2, 0)
                    
            norm_img = (img + 1) / 2 * 255
            norm_img = norm_img.clip(0, 255)
            
            q2_images[i] = norm_img
            
        half_pos = img_size // 2
        q2_images[:, half_pos:, :, :] = 0

        sep_images = np.zeros((B * N, img_size, img_size, 3), dtype=np.uint8)
        sep_images[:, :, :] = q2_images

        sep_images = add_horizontal_lines(sep_images, num_rows=2)

        images = np.zeros((B * 2, img_size, img_size, 3), dtype=np.uint8)

        num_aggregate_q2_images = B * N // 2
        aggregate_q2_images = np.zeros((num_aggregate_q2_images, img_size, img_size, 3), dtype=np.uint8)
        
        for prompt_idx in range(num_aggregate_q2_images):
            halfs = q2_images[2*prompt_idx:2*(prompt_idx+1)]
            for i in range(2):
                aggregate_q2_images[prompt_idx,half_pos*i:half_pos*(i+1),:,:] = halfs[i, :half_pos, :, :]

        aggregate_q2_images = add_horizontal_lines(aggregate_q2_images, num_rows=2)
        
        img_idx = 0
        outputs = []
        base64_input_image = encode_pil_image(Image.fromarray(np.array(batch['_input_img'])[0]))
        for img_idx in range(num_aggregate_q2_images):
            base64_output_image = encode_pil_image(Image.fromarray(aggregate_q2_images[img_idx]))
            response = client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": gpt_select_second_row_prompt.format(prompts[0])},
                            # (A) Source image 
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_input_image}",
                            },
                            # (B) Composite (4 quarters) 
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_output_image}",
                            },
                        ],
                    }
                ],
            )
            outputs.append(response.output_text)
        outputs = [o.lower() for o in outputs]
        save_text_record(records_dir, f"{sample_ids}-half_select.txt", outputs)

        gen_tokens, sep_images = propagate_selection_flexible(
            gen_tokens,
            verify_texts=outputs,
            batch_size=B,
            N=N,
            grouped_by_4=args.grouped_by_4,
            images=sep_images,
            aggregate_images=aggregate_q2_images,
            record_dir=records_dir,
            sample_ids=sample_ids,
            prompts_per_group=2,
        )

        sep_pil_images = [Image.fromarray(sep_image) for sep_image in sep_images]
        outputs = []
        base64_input_image = encode_pil_image(Image.fromarray(np.array(batch['_input_img'])[0]))
        for img_idx in range(N):
            base64_output_image = encode_pil_image(sep_pil_images[img_idx])
            response = client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": refine_prompt.format(prompts[0])},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_input_image}",
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_output_image}",
                            },
                        ],
                    }
                ],
            )
            outputs.append(response.output_text)
        outputs = [s[(s.find('"') + 1):s.rfind('"')] for s in outputs]

        save_text_record(records_dir, f"{sample_ids}-refined.txt", outputs)

        input_txt_embs = tokenize_text(
            t5_model.tokenizer, t5_model, outputs, device
        )
        input_txt_embs = input_txt_embs.repeat(B * N, 1, 1)
        with torch.inference_mode():
            final_generated_tokens = grid_generate_edit_third_stage(
                gpt_model, input_txt_embs, input_img_indices, input_mode,
                gen_tokens=gen_tokens,
                image_token_num_per_image=latent_size ** 2,
                emb_masks=None,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True,
            )
            
        qz_shape = [1, 8, latent_size, latent_size]
        
        dec = np.zeros((B * N, img_size, img_size, 3), dtype=np.uint8)
        
        for i, image_token in enumerate(final_generated_tokens):
            img = vq_model.decode_code([image_token], qz_shape)[0]
            img = img.to(torch.float32).cpu().numpy().transpose(1, 2, 0)
                    
            norm_img = (img + 1) / 2 * 255
            norm_img = norm_img.clip(0, 255)
            
            dec[i] = norm_img
        images = np.zeros((B * N, img_size, img_size, 3), dtype=np.uint8)
        images[:, :, :] = dec
        pil_images = [Image.fromarray(image) for image in images]
        
        src_image = Image.fromarray(np.uint8(batch["_input_img"][0].cpu().numpy()))
        src_image.save(os.path.join(visualization_dir, f"{sample_ids}_src.png"))
        for idx in range(B * N):
            pil_images[idx].save(os.path.join(visualization_dir, f"{sample_ids}_tgt_{idx}.png"))
        
        progress_bar.update(1)
    progress_bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill-mode", type=str, choices=['dinov2', 'clip', 'clipseg'], default=None)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=DEFAULT_VQ_CKPT_PATH, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=DEFAULT_GPT_CKPT_PATH, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 'edit'], default="edit")
    parser.add_argument("--gpt-mode", type=str, choices=['img_cls_emb', 'joint_cls_emb'], default='joint_cls_emb')
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--t5-path", type=str, default=DEFAULT_T5_PATH)
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="output dir to save results")
    parser.add_argument("--output-name", type=str, default='GridAR__EditAR__PIE__N4', help="output name to save results")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--sample-index", type=str, default=None)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--grouped-by-4", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
