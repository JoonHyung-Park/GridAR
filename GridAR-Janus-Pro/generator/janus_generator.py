from dataclasses import replace
from typing import Optional

import numpy as np
import torch

import decode

from generator.base_generator import BaseGenerator, T2IRequest


class JanusGenerator(BaseGenerator):
    def __init__(self, mmgpt, max_batch_size: int):
        super().__init__(device=mmgpt.device)
        self.mmgpt = mmgpt
        self.max_batch_size = max_batch_size

    def _assert_ready(self, req: T2IRequest) -> None:
        if req.top_k is not None or req.top_p is not None:
            raise ValueError("JanusGenerator only supports multinomial sampling without top-k/top-p")

    def _pad_after_first_quarter(self, tokens: torch.Tensor, quarter: int, mode: str) -> torch.Tensor:
        if mode == "zero":
            tokens[:, quarter:] = 0
        elif mode == "mean":
            tokens[:, quarter:] = tokens[:, :quarter].float().mean(dim=1, keepdim=True).long()
        elif mode == "repeat":
            tokens[:, quarter:2 * quarter] = tokens[:, :quarter]
            tokens[:, 2 * quarter:3 * quarter] = tokens[:, :quarter]
            tokens[:, 3 * quarter:4 * quarter] = tokens[:, :quarter]
        else:
            raise ValueError(f"Unsupported pad mode: {mode}")
        return tokens

    def _pad_after_second_quarter(self, tokens: torch.Tensor, half: int, mode: str) -> torch.Tensor:
        if mode == "zero":
            tokens[:, half:] = 0
        elif mode == "mean":
            tokens[:, half:] = tokens[:, :half].float().mean(dim=1, keepdim=True).long()
        elif mode == "repeat":
            tokens[:, half:] = tokens[:, :half]
        else:
            raise ValueError(f"Unsupported pad mode: {mode}")
        return tokens

    def _use_3way_cfg(self, req: T2IRequest, stage_key: str) -> bool:
        stage_cfg = getattr(req.cfg.method, stage_key, None)
        return bool(stage_cfg and getattr(stage_cfg, "use_3way_cfg", False))

    def _prefill_chain(self, req: T2IRequest, given_tokens: Optional[torch.Tensor], pair_dim: int):
        inputs_embeds = req.inputs_embeds
        attention_mask = req.attention_mask

        if given_tokens is not None:
            pair_tokens = given_tokens.repeat_interleave(pair_dim, 0)
            pair_embeds = self.mmgpt.prepare_gen_img_embeds(pair_tokens)
            pair_mask = torch.ones_like(pair_embeds[..., 0], dtype=attention_mask.dtype)
            row_prompt = torch.cat([inputs_embeds, pair_embeds], dim=1)
            row_mask = torch.cat([attention_mask, pair_mask], dim=1)
        else:
            row_prompt = inputs_embeds
            row_mask = attention_mask

        outputs = self.mmgpt.language_model.model(
            inputs_embeds=row_prompt,
            attention_mask=row_mask,
            use_cache=True,
            past_key_values=None,
            output_hidden_states=False,
        )
        return outputs, row_mask

    def _append_step(self, row_mask: torch.Tensor, past_key_values, pair_embeds: torch.Tensor):
        new_mask = torch.cat(
            [
                row_mask,
                torch.ones((row_mask.size(0), pair_embeds.size(1)), dtype=row_mask.dtype, device=row_mask.device),
            ],
            dim=1,
        )
        outputs = self.mmgpt.language_model.model(
            inputs_embeds=pair_embeds,
            attention_mask=new_mask,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=False,
        )
        return outputs, new_mask

    @torch.inference_mode()
    def _chain_2way_core(self, req: T2IRequest, given_tokens: Optional[torch.Tensor], steps: int) -> torch.Tensor:
        batch_twice = req.inputs_embeds.size(0)
        if batch_twice % 2 != 0:
            raise ValueError("2-way generation expects a batch size divisible by 2")

        batch_size = batch_twice // 2
        device = req.inputs_embeds.device
        decode_fn = getattr(decode, req.cfg.decode_func)
        temperature = req.temperature

        outputs, row_mask = self._prefill_chain(req, given_tokens, pair_dim=2)
        past_key_values = outputs.past_key_values

        generated = torch.zeros((batch_size, steps), dtype=torch.long, device=device)
        for step in range(steps):
            logits = self.mmgpt.gen_head(outputs.last_hidden_state[:, -1, :])
            uncond_logits, cond_logits = logits[0::2, :], logits[1::2, :]
            fused_logits = decode_fn(logit_cond=cond_logits, logit_uncond=uncond_logits, scale=req.cfg.cfg_scale)
            probs = torch.softmax(fused_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated[:, step] = next_token.squeeze(-1)

            pair_tokens = next_token.repeat_interleave(2, 0).squeeze(-1)
            pair_embeds = self.mmgpt.prepare_gen_img_embeds(pair_tokens).unsqueeze(1)
            outputs, row_mask = self._append_step(row_mask, past_key_values, pair_embeds)
            past_key_values = outputs.past_key_values

        return generated

    @torch.inference_mode()
    def _chain_3way_core(self, req: T2IRequest, given_tokens: Optional[torch.Tensor], steps: int) -> torch.Tensor:
        batch_triple = req.inputs_embeds.size(0)
        if batch_triple % 3 != 0:
            raise ValueError("3-way generation expects a batch size divisible by 3")

        batch_size = batch_triple // 3
        device = req.inputs_embeds.device
        decode_fn = getattr(decode, req.cfg.method.decode_func)
        temperature = req.temperature

        outputs, row_mask = self._prefill_chain(req, given_tokens, pair_dim=3)
        past_key_values = outputs.past_key_values

        generated = torch.zeros((batch_size, steps), dtype=torch.long, device=device)
        for step in range(steps):
            logits = self.mmgpt.gen_head(outputs.last_hidden_state[:, -1, :])
            uncond_logits = logits[0::3, :]
            cond_logits = logits[1::3, :]
            refined_logits = logits[2::3, :]
            fused_logits = decode_fn(
                logit_cond_modified=refined_logits,
                logit_cond=cond_logits,
                logit_uncond=uncond_logits,
                scale=req.cfg.cfg_scale,
            )
            probs = torch.softmax(fused_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated[:, step] = next_token.squeeze(-1)

            pair_tokens = next_token.repeat_interleave(3, 0).squeeze(-1)
            pair_embeds = self.mmgpt.prepare_gen_img_embeds(pair_tokens).unsqueeze(1)
            outputs, row_mask = self._append_step(row_mask, past_key_values, pair_embeds)
            past_key_values = outputs.past_key_values

        return generated

    @torch.inference_mode()
    def _chain_2way(self, req: T2IRequest, given_tokens: Optional[torch.Tensor], steps: int) -> torch.Tensor:
        batch_size = req.inputs_embeds.size(0) // 2
        if not self.max_batch_size or batch_size <= self.max_batch_size:
            return self._chain_2way_core(req, given_tokens, steps)

        parts = []
        for batch_start in range(0, batch_size, self.max_batch_size):
            batch_end = min(batch_size, batch_start + self.max_batch_size)
            pair_slice = slice(2 * batch_start, 2 * batch_end)
            sub_request = replace(
                req,
                inputs_embeds=req.inputs_embeds[pair_slice],
                attention_mask=req.attention_mask[pair_slice] if req.attention_mask is not None else None,
            )
            sub_given_tokens = given_tokens[batch_start:batch_end] if given_tokens is not None else None
            parts.append(self._chain_2way_core(sub_request, sub_given_tokens, steps))
        return torch.cat(parts, dim=0)

    @torch.inference_mode()
    def _chain_3way(self, req: T2IRequest, given_tokens: Optional[torch.Tensor], steps: int) -> torch.Tensor:
        batch_size = req.inputs_embeds.size(0) // 3
        if not self.max_batch_size or batch_size <= self.max_batch_size:
            return self._chain_3way_core(req, given_tokens, steps)

        parts = []
        for batch_start in range(0, batch_size, self.max_batch_size):
            batch_end = min(batch_size, batch_start + self.max_batch_size)
            pair_slice = slice(3 * batch_start, 3 * batch_end)
            sub_request = replace(
                req,
                inputs_embeds=req.inputs_embeds[pair_slice],
                attention_mask=req.attention_mask[pair_slice] if req.attention_mask is not None else None,
            )
            sub_given_tokens = given_tokens[batch_start:batch_end] if given_tokens is not None else None
            parts.append(self._chain_3way_core(sub_request, sub_given_tokens, steps))
        return torch.cat(parts, dim=0)

    @torch.inference_mode()
    def generate_t2i_first_quarter(self, req: T2IRequest) -> torch.Tensor:
        self._assert_ready(req)
        total_tokens = int(req.image_token_num_per_image)
        quarter = total_tokens // 4
        first_quarter = self._chain_2way(req, given_tokens=None, steps=quarter)

        output = torch.zeros((first_quarter.size(0), total_tokens), dtype=torch.long, device=first_quarter.device)
        output[:, :quarter] = first_quarter
        return self._pad_after_first_quarter(output, quarter, req.pad)

    @torch.inference_mode()
    def generate_t2i_second_quarter(self, req: T2IRequest, gen_tokens_q1: torch.Tensor) -> torch.Tensor:
        self._assert_ready(req)
        total_tokens = int(req.image_token_num_per_image)
        quarter = total_tokens // 4
        half = total_tokens // 2
        past_q1 = gen_tokens_q1[:, :quarter].contiguous()

        if self._use_3way_cfg(req, "after_first_quarter"):
            second_quarter = self._chain_3way(req, past_q1, quarter)
        else:
            second_quarter = self._chain_2way(req, past_q1, quarter)

        output = torch.zeros((past_q1.size(0), total_tokens), dtype=torch.long, device=past_q1.device)
        output[:, :quarter] = past_q1
        output[:, quarter:half] = second_quarter
        return self._pad_after_second_quarter(output, half, req.pad)

    @torch.inference_mode()
    def generate_t2i_second_half(self, req: T2IRequest, gen_tokens_half: torch.Tensor) -> torch.Tensor:
        self._assert_ready(req)
        total_tokens = int(req.image_token_num_per_image)
        half = total_tokens // 2
        past_half = gen_tokens_half[:, :half].contiguous()

        if self._use_3way_cfg(req, "after_first_half"):
            remaining = self._chain_3way(req, past_half, total_tokens - half)
        else:
            remaining = self._chain_2way(req, past_half, total_tokens - half)

        output = torch.zeros((past_half.size(0), total_tokens), dtype=torch.long, device=past_half.device)
        output[:, :half] = past_half
        output[:, half:] = remaining
        return output

    @torch.inference_mode()
    def generate_t2i_stage_prefix_preserving(self, req: T2IRequest) -> torch.Tensor:
        self._assert_ready(req)
        first_quarter = self.generate_t2i_first_quarter(req)
        first_half = self.generate_t2i_second_quarter(req, gen_tokens_q1=first_quarter)
        return self.generate_t2i_second_half(req, gen_tokens_half=first_half)

    @torch.inference_mode()
    def image_decode(self, req: T2IRequest, token_ids: torch.Tensor) -> np.ndarray:
        decoded = self.mmgpt.gen_vision_model.decode_code(
            token_ids.to(dtype=torch.int),
            shape=[token_ids.size(0), 8, req.img_size // req.patch_size, req.img_size // req.patch_size],
        )
        decoded = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        return np.clip((decoded + 1) / 2 * 255, 0, 255)
