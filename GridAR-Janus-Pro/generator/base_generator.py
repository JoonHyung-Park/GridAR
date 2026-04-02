from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class T2IRequest:
    inputs_embeds: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    image_token_num_per_image: int = 576
    img_size: int = 384
    patch_size: int = 16
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    pad: str = "repeat"
    cfg: Optional[Any] = None

class BaseGenerator(ABC):
    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    @torch.inference_mode()
    def generate_t2i_first_quarter(self, req: T2IRequest) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def generate_t2i_second_quarter(self, req: T2IRequest, gen_tokens_q1: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def generate_t2i_second_half(self, req: T2IRequest, gen_tokens_half: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def generate_t2i_stage_prefix_preserving(self, req: T2IRequest) -> torch.Tensor:
        """Generate a full image using the same quarter/half stage boundaries as GridAR."""
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def image_decode(self, req: T2IRequest, token_ids: torch.Tensor) -> np.ndarray:
        raise NotImplementedError
