# GridAR generation helpers built on top of upstream EditAR generation.
import torch
import torch._dynamo.config
import torch._inductor.config
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
from editar.autoregressive.models.generate_edit import decode_n_tokens, prefill

@torch.no_grad()
def generate_edit_quarter(model, 
        input_txt_embs, 
        input_img_indices, 
        input_mode, 
        image_token_num_per_image: int = 1024, 
        emb_masks=None, 
        cfg_scale=1.0, 
        cfg_interval=-1, 
        pad: str = "zero",
        **sampling_kwargs
    ):
    if model.model_type == 'edit':
        if cfg_scale > 1.0:
            input_txt_null = torch.zeros_like(input_txt_embs) + model.cls_embedding.uncond_embedding
            input_txt_combined = torch.cat([input_txt_embs, input_txt_null])
            input_img_combined = torch.cat([input_img_indices, input_img_indices])
            input_mode_combined = torch.cat([input_mode, input_mode])
        else:
            input_txt_combined = input_txt_embs
            input_img_combined = input_img_indices
            input_mode_combined = input_mode
        T = input_txt_combined.shape[1] + input_img_combined.shape[1]
    else:
        raise Exception("please check model type")

    quarter = image_token_num_per_image // 4
    
    T_new = T + quarter
    max_seq_length = T_new
    max_batch_size = input_txt_embs.shape[0]
    
    device = input_img_indices.device
    generated_tokens = torch.zeros((max_batch_size, quarter), dtype=torch.int, device=device)

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
   
    # create an empty tensor of the expected final shape and fill in the current tokens

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, input_txt_combined, input_img_combined, input_mode_combined, input_pos, cfg_scale, **sampling_kwargs)
    generated_tokens[:, 0:1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    n_tokens, _ = decode_n_tokens(model, next_token, input_pos, quarter-1, cfg_scale, cfg_interval, **sampling_kwargs)
    generated_tokens[:, 1:] = torch.cat(n_tokens, dim=1)

    image_tokens = torch.zeros((max_batch_size, image_token_num_per_image), dtype=torch.int).cuda()
    image_tokens[:, :quarter] = generated_tokens
    
    if pad == "zero":
        image_tokens[:, quarter:] = 0

    elif pad == "mean":
        mean_vals = image_tokens[:, :quarter].float().mean(dim=1, keepdim=True)
        image_tokens[:, quarter:] = mean_vals.long()

    elif pad == "repeat":
        image_tokens[:, quarter:(2*quarter)] = image_tokens[:, :quarter]
        image_tokens[:, (2*quarter):(3*quarter)] = image_tokens[:, :quarter]
        image_tokens[:, (3*quarter):(4*quarter)] = image_tokens[:, :quarter]

    return image_tokens


@torch.no_grad()
def generate_edit_second_quarter(model, 
        input_txt_embs, 
        input_img_indices, 
        input_mode, 
        gen_tokens,
        image_token_num_per_image: int = 1024, 
        emb_masks=None, 
        cfg_scale=1.0, 
        cfg_interval=-1, 
        pad: str = "zero",
        **sampling_kwargs
    ):
    if not model.model_type == 'edit':
        raise Exception("please check model type")

    T = input_txt_embs.shape[1] + input_img_indices.shape[1]
    quarter = image_token_num_per_image // 4

    T_new = T + quarter * 2
    max_seq_length = T_new
    max_batch_size = input_txt_embs.shape[0]
    device = input_img_indices.device

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
   
    past_q1_ids = gen_tokens[:, :quarter].to(device).long()

    if cfg_scale > 1.0:
        input_txt_null = torch.zeros_like(input_txt_embs) + model.cls_embedding.uncond_embedding
        input_txt_combined = torch.cat([input_txt_embs, input_txt_null])
        input_img_combined = torch.cat([input_img_indices, input_img_indices])
        input_mode_combined = torch.cat([input_mode, input_mode])
        idx = torch.cat([past_q1_ids, past_q1_ids], dim=0)[:, :-1] # (2B, quarter-1)
    else:
        input_txt_combined = input_txt_embs # (B, T_text, caption_dim)
        input_img_combined = input_img_indices
        input_mode_combined = input_mode
        idx = past_q1_ids[:, :-1] # (B, quarter-1)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, input_txt_combined, input_img_combined, input_mode_combined, input_pos, cfg_scale, **sampling_kwargs)
    
    input_pos = torch.arange(T, T + quarter - 1, device=device)
    _ = model(input_txt_embs=None, input_img_indices=None, edited_img_indices=idx, input_mode=input_mode_combined, input_pos=input_pos)

    start_token = past_q1_ids[:, -1:].contiguous().long()                         # (B,1)
    input_pos   = torch.tensor([T + quarter - 1], device=device, dtype=torch.int)

    n_tokens, _ = decode_n_tokens(model, start_token, input_pos, int(quarter), cfg_scale, cfg_interval, **sampling_kwargs)
    q2 = torch.cat(n_tokens, dim=1).long() 
    
    out = gen_tokens.clone().to(device).long()
    out[:, quarter:2*quarter] = q2
    half = image_token_num_per_image // 2
    if pad == "zero":
        out[:, half:] = 0
    elif pad == "mean":
        mean_vals = out[:, :half].float().mean(dim=1, keepdim=True)
        out[:, half:] = mean_vals.long()
    elif pad == "repeat":
        out[:, half:] = out[:, :half]
        
    return out

@torch.no_grad()
def grid_generate_edit_third_stage(model, 
        input_txt_embs, 
        input_img_indices, 
        input_mode, 
        gen_tokens,
        image_token_num_per_image: int = 1024, 
        emb_masks=None, 
        cfg_scale=1.0, 
        cfg_interval=-1, 
        pad: str = "zero",
        **sampling_kwargs
    ):
    if not model.model_type == 'edit':
        raise Exception("please check model type")

    T = input_txt_embs.shape[1] + input_img_indices.shape[1]
    half = image_token_num_per_image // 2

    T_new = T + image_token_num_per_image
    max_seq_length = T_new
    max_batch_size = input_txt_embs.shape[0]
    
    device = input_img_indices.device

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
   
    past_half_ids = gen_tokens[:, :half].to(device).long()

    if cfg_scale > 1.0:
        input_txt_null = torch.zeros_like(input_txt_embs) + model.cls_embedding.uncond_embedding
        input_txt_combined = torch.cat([input_txt_embs, input_txt_null])
        input_img_combined = torch.cat([input_img_indices, input_img_indices])
        input_mode_combined = torch.cat([input_mode, input_mode])
        idx = torch.cat([past_half_ids, past_half_ids], dim=0)[:, :-1] # (2B, half-1)
    else:
        input_txt_combined = input_txt_embs # (B, T_text, caption_dim)
        input_img_combined = input_img_indices
        input_mode_combined = input_mode
        idx = past_half_ids[:, :-1] # (B, half-1)

    input_pos = torch.arange(0, T, device=device)
    _ = prefill(model, input_txt_combined, input_img_combined, input_mode_combined, input_pos, cfg_scale, **sampling_kwargs)
    
    input_pos = torch.arange(T, T + half - 1, device=device)
    _ = model(input_txt_embs=None, input_img_indices=None, edited_img_indices=idx, input_mode=input_mode_combined, input_pos=input_pos)

    start_token = past_half_ids[:, -1:].contiguous().long()                         # (B,1)
    input_pos   = torch.tensor([T + half - 1], device=device, dtype=torch.int)

    n_tokens, _ = decode_n_tokens(model, start_token, input_pos, int(half), cfg_scale, cfg_interval, **sampling_kwargs)
    
    last_half = torch.cat(n_tokens, dim=1).long() 
    
    out = gen_tokens.clone().to(device).long()
    out[:, half:] = last_half
        
    return out
