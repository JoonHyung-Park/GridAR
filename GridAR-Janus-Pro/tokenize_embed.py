from typing import List

import torch

from janus.models import MultiModalityCausalLM, VLChatProcessor



def _build_conversation(prompt: str, model_name: str) -> list[dict[str, str]]:
    if "Janus-Pro" in model_name:
        user_role = "<|User|>"
        assistant_role = "<|Assistant|>"
    else:
        user_role = "User"
        assistant_role = "Assistant"

    return [
        {"role": user_role, "content": prompt},
        {"role": assistant_role, "content": ""},
    ]



def _batchify_prompt_ids(vl_chat_processor: VLChatProcessor, prompts: List[str], model_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_ids: list[torch.Tensor] = []
    max_seq_len = 0

    for prompt in prompts:
        conversation = _build_conversation(prompt, model_name)
        sft_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt_text = sft_prompt + vl_chat_processor.image_start_tag
        input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(prompt_text))
        sequence_ids.append(input_ids)
        max_seq_len = max(max_seq_len, input_ids.shape[0])

    batch_size = len(sequence_ids)
    batched_input_ids = torch.full((batch_size, max_seq_len), vl_chat_processor.pad_id).long()
    batched_attention_mask = torch.zeros((batch_size, max_seq_len)).long()

    for index, input_ids in enumerate(sequence_ids):
        seq_len = input_ids.shape[0]
        batched_input_ids[index, -seq_len:] = input_ids
        batched_attention_mask[index, -seq_len:] = 1

    return batched_input_ids, batched_attention_mask



def tokenize_text_janus(
    vl_chat_processor: VLChatProcessor,
    embedding_model: MultiModalityCausalLM,
    prompts: List[str],
    model_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batched_input_ids, batched_attention_mask = _batchify_prompt_ids(vl_chat_processor, prompts, model_name)
    batch_size, max_seq_len = batched_input_ids.shape

    tokens = torch.zeros((batch_size * 2, max_seq_len), dtype=torch.long)
    attention_mask = batched_attention_mask.repeat_interleave(2, dim=0)
    tokens[0::2, 1:-1] = vl_chat_processor.pad_id
    tokens[1::2] = batched_input_ids

    for index in range(batch_size):
        first_one = (batched_attention_mask[index] == 1).nonzero(as_tuple=True)[0][0]
        tokens[2 * index, first_one] = batched_input_ids[index, first_one]
        tokens[2 * index, -1] = batched_input_ids[index, -1]

    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)
    input_embeds_prompt = embedding_model.language_model.get_input_embeddings()(tokens)
    return input_embeds_prompt, attention_mask



def tokenize_text_janus_3_way(
    vl_chat_processor: VLChatProcessor,
    embedding_model: MultiModalityCausalLM,
    prompts: List[str],
    modified_prompts: List[str],
    model_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(prompts) != len(modified_prompts):
        raise ValueError("prompts and modified_prompts must have the same length")

    batched_input_ids_orig, batched_attention_mask_orig = _batchify_prompt_ids(vl_chat_processor, prompts, model_name)
    batched_input_ids_mod, batched_attention_mask_mod = _batchify_prompt_ids(vl_chat_processor, modified_prompts, model_name)

    if batched_input_ids_orig.shape[1] != batched_input_ids_mod.shape[1]:
        max_seq_len = max(batched_input_ids_orig.shape[1], batched_input_ids_mod.shape[1])
        batch_size = batched_input_ids_orig.shape[0]

        def _pad(batch_ids: torch.Tensor, batch_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            padded_ids = torch.full((batch_size, max_seq_len), vl_chat_processor.pad_id).long()
            padded_mask = torch.zeros((batch_size, max_seq_len)).long()
            padded_ids[:, -batch_ids.shape[1]:] = batch_ids
            padded_mask[:, -batch_mask.shape[1]:] = batch_mask
            return padded_ids, padded_mask

        batched_input_ids_orig, batched_attention_mask_orig = _pad(batched_input_ids_orig, batched_attention_mask_orig)
        batched_input_ids_mod, batched_attention_mask_mod = _pad(batched_input_ids_mod, batched_attention_mask_mod)

    batch_size, max_seq_len = batched_input_ids_orig.shape
    tokens = torch.full((batch_size * 3, max_seq_len), vl_chat_processor.pad_id, dtype=torch.long)
    tokens[1::3] = batched_input_ids_orig
    tokens[2::3] = batched_input_ids_mod

    for index in range(batch_size):
        first_one = (batched_attention_mask_orig[index] == 1).nonzero(as_tuple=True)[0][0]
        tokens[3 * index, first_one] = batched_input_ids_orig[index, first_one]
        tokens[3 * index, -1] = batched_input_ids_orig[index, -1]

    attention_mask = torch.zeros((batch_size * 3, max_seq_len), dtype=torch.long)
    attention_mask[0::3] = batched_attention_mask_orig
    attention_mask[1::3] = batched_attention_mask_orig
    attention_mask[2::3] = batched_attention_mask_mod

    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)
    input_embeds_prompt = embedding_model.language_model.get_input_embeddings()(tokens)
    return input_embeds_prompt, attention_mask
