# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import threading

# ----------------
import math
import logging
import functools
from functools import partial
from typing import Callable, Final, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


logger = logging.getLogger(__name__)

# ------------------------------- Small utils ----------------------------------


def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if we can use PyTorch fused scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")


class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing = False

    _gradient_checkpointing_func = checkpoint

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            # self._gradient_checkpointing_func must be non-reentrant variant of checkpoint
            # use_reentrant=False is required, see `use_reentrant` doc for more details
            # https://docs.pytorch.org/docs/2.9/checkpoint.html#torch.utils.checkpoint.checkpoint
            return self._gradient_checkpointing_func(super().__call__, *args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)


# ---------------------------------- MLP --------------------------------------


class Mlp(nn.Module):
    """
    MLP used in ViT-style blocks.

    Supports Linear or 1x1 Conv 'linear_layer' for token/channel mixing.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, C] for Linear variant; caller is responsible for shapes.
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -------------------------------- Attention ----------------------------------


class Attention(nn.Module):
    """
    Multi-Head Self-Attention with optional fused SDPA fallback.

    If PyTorch provides `scaled_dot_product_attention`, it will be used
    (usually faster and more stable); otherwise we use a manual implementation.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, T, C]
            Input sequence.

        Returns
        -------
        Tensor, shape [B, T, C]
            Output sequence after MHSA + projection.
        """
        B, T, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # 3 x [B, H, T, Dh]
        )
        q, k, v = qkv.unbind(0)  # each: [B, H, T, Dh]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )  # [B, H, T, Dh]
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, H, T, T]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, H, T, Dh]

        x = x.transpose(1, 2).reshape(B, T, C)  # [B, T, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------- Utilities -----------------------------------


def basic_init(module: nn.Module) -> None:
    """
    Apply a basic initialization scheme to Linear layers.

    - Weight: Xavier uniform initialization.
    - Bias: Set to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Parameters
    ----------
    t : torch.Tensor
        Shape [B]. Each element is a timestep index, may be fractional.
    dim : int
        Dimensionality of the output embedding.
    max_period : int, default=100
        Controls the minimum frequency of the sinusoids.

    Returns
    -------
    torch.Tensor
        Shape [B, dim]. Sinusoidal embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device) / half)
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ------------------------------- Core Layers ----------------------------------


######### Feature-Aware MoE Decoder #########
class ActionMoEDecoder(nn.Module):
    def __init__(
        self,
        base_decoder: nn.Linear,
        expert_names: list[str] | None = None,
        router_enabled: bool = True,
        router_select_mode: str = "hard",
        routing_basis_dim: int = -1,
    ) -> None:
        super().__init__()
        if not isinstance(base_decoder, nn.Linear):
            raise TypeError(f"base_decoder must be nn.Linear, got {type(base_decoder)!r}")
        if router_select_mode != "hard":
            raise NotImplementedError(f"Unsupported router_select_mode: {router_select_mode}")

        self.base_decoder = base_decoder
        self.router_enabled = router_enabled
        self.router_select_mode = router_select_mode
        self.routing_basis_dim = routing_basis_dim if routing_basis_dim > 0 else self.dim_action
        self.expert_decoders = nn.ModuleDict()
        self.expert_names: list[str] = []
        self.active_expert_name: str | None = None
        self._routing_buffer_names: dict[str, str] = {}

        for expert_name in expert_names or []:
            self.append_expert(expert_name)

    @property
    def hidden_size(self) -> int:
        return self.base_decoder.in_features

    @property
    def dim_action(self) -> int:
        return self.base_decoder.out_features

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return "".join(char if char.isalnum() or char == "_" else "_" for char in name)

    def _routing_buffer_name(self, expert_name: str) -> str:
        if expert_name not in self._routing_buffer_names:
            self._routing_buffer_names[expert_name] = f"routing_basis_{self._sanitize_name(expert_name)}"
        return self._routing_buffer_names[expert_name]

    def get_expert_weight_name(self, expert_name: str) -> str:
        return f"experts.{expert_name}.weight"

    def append_expert(self, expert_name: str) -> nn.Linear:
        if expert_name in self.expert_decoders:
            return self.expert_decoders[expert_name]

        expert = nn.Linear(self.hidden_size, self.dim_action)
        nn.init.zeros_(expert.weight)  # TODO
        if expert.bias is not None:
            nn.init.zeros_(expert.bias)
        self.expert_decoders[expert_name] = expert
        self.expert_names.append(expert_name)

        buffer_name = self._routing_buffer_name(expert_name)
        routing_basis = torch.zeros(self.hidden_size, self.routing_basis_dim)
        self.register_buffer(buffer_name, routing_basis, persistent=True)
        # logger.info(f"[[Model]] Added expert '{expert_name}' to ActionMoEDecoder with routing buffer '{buffer_name}' of shape {routing_basis.shape}")
        return expert

    def set_active_expert(self, expert_name: str | None) -> None:
        if expert_name is not None and expert_name not in self.expert_decoders:
            raise KeyError(f"Unknown decoder expert: {expert_name}")
        self.active_expert_name = expert_name

    def update_routing_basis(self, expert_name: str, routing_basis: torch.Tensor) -> None:
        if expert_name not in self.expert_decoders:
            raise KeyError(f"Unknown decoder expert: {expert_name}")

        if routing_basis.ndim != 2:
            raise ValueError(f"routing_basis must be 2D, got shape {tuple(routing_basis.shape)}")
        if routing_basis.shape[0] != self.hidden_size:
            raise ValueError(
                f"routing_basis first dim must match hidden_size={self.hidden_size}, got {routing_basis.shape[0]}"
            )
        if routing_basis.shape[1] != self.routing_basis_dim:
            raise ValueError(
                f"routing_basis second dim must match routing_basis_dim={self.routing_basis_dim}, got {routing_basis.shape[1]}"
            )

        routing_basis = routing_basis.detach()
        buffer_name = self._routing_buffer_name(expert_name)
        assert hasattr(self, buffer_name), f"Missing routing buffer for expert {expert_name}: {buffer_name}"
        buffer = getattr(self, buffer_name)
        buffer.data = routing_basis.to(device=buffer.device, dtype=buffer.dtype)
        logger.info(
            f"[[Model]] Updated routing basis for expert '{expert_name}' in buffer '{buffer_name}' with new basis of shape {routing_basis.shape}"
        )

    def _load_confuse_matrix(self) -> dict:
        if not os.path.exists(self.confuse_matrix_file):
            raise FileNotFoundError(f"Confusion matrix file does not exist: {self.confuse_matrix_file}")
        with open(self.confuse_matrix_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Confusion matrix JSON must be an object, got {type(data)!r}")
        return data

    def _write_confuse_matrix(self, data: dict) -> None:
        tmp_file = f"{self.confuse_matrix_file}.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, self.confuse_matrix_file)
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def _accumulate_confuse_matrix(self, selected_expert_index: torch.Tensor) -> None:
        if selected_expert_index.ndim != 1:
            raise ValueError(
                f"selected_expert_index must be 1D with shape [B], got {tuple(selected_expert_index.shape)}"
            )

        route_counts: dict[str, int] = {}
        for raw_idx in selected_expert_index.detach().to(device="cpu", dtype=torch.long).tolist():
            expert_idx = int(raw_idx)
            if expert_idx < 0 or expert_idx >= len(self.expert_names):
                raise IndexError(f"Routed expert index out of range: {expert_idx}")
            expert_name = self.expert_names[expert_idx]
            route_counts[expert_name] = route_counts.get(expert_name, 0) + 1

        with self._confuse_file_lock:
            data = self._load_confuse_matrix()
            if self.curr_skill not in data:
                raise KeyError(f"Skill '{self.curr_skill}' missing in confusion matrix file {self.confuse_matrix_file}")

            row = data[self.curr_skill]
            if not isinstance(row, dict):
                raise ValueError(
                    f"Confusion matrix row for skill '{self.curr_skill}' must be an object, got {type(row)!r}"
                )

            for expert_name, increment in route_counts.items():
                if expert_name not in row:
                    raise KeyError(
                        f"Expert '{expert_name}' missing in confusion row '{self.curr_skill}' of {self.confuse_matrix_file}"
                    )
                try:
                    current_value = int(row[expert_name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Confusion value at ['{self.curr_skill}']['{expert_name}'] must be int-like, got {row[expert_name]!r}"
                    ) from exc
                row[expert_name] = current_value + increment

            self._write_confuse_matrix(data)

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        if not self.router_enabled:
            raise RuntimeError("Routing is disabled; _route should not be called")
        if len(self.expert_names) == 0:
            raise RuntimeError("No decoder experts registered")
        assert self.active_expert_name is None, "_route should not be called when an active expert is set"

        routing_bases = [getattr(self, self._routing_buffer_name(expert_name)) for expert_name in self.expert_names]
        assert all([basis.shape[1] == self.routing_basis_dim for basis in routing_bases])
        routing_bases = torch.stack(
            [routing_basis.to(device=x.device, dtype=x.dtype) for routing_basis in routing_bases], dim=0
        )

        # b: batch_size s: seq_len h: hidden_size e: num_decoders k: num_bases
        # || x * N_t ||^2  for t-th skill
        projected = torch.einsum("bsh,ehk->besk", x, routing_bases)
        score_tensor = projected.pow(2).sum(dim=(2, 3))
        selected_expert_index = score_tensor.argmax(dim=-1)
        # debug
        # if not self.training:
        #     self._accumulate_confuse_matrix(selected_expert_index)
        return selected_expert_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_decoder(x)
        if not self.router_enabled:
            return base_output
        if len(self.expert_names) == 0:
            raise RuntimeError("No decoder experts registered")

        if self.active_expert_name is not None:
            expert_output = self.expert_decoders[self.active_expert_name](x)
            return base_output + expert_output

        selected_expert_index = self._route(x)  # shape: [B]
        # selected_expert_index = torch.tensor([2] * x.shape[0], dtype=torch.long, device=x.device)
        selected_output = torch.zeros_like(base_output)  # shape: [B, dim_action]
        for expert_index, expert_name in enumerate(self.expert_names):
            expert_mask = selected_expert_index == expert_index
            if not torch.any(expert_mask).item():
                continue
            selected_output[expert_mask] = self.expert_decoders[expert_name](x[expert_mask])
        return base_output + selected_output


class DomainAwareLinear(nn.Module):
    """
    Linear layer with domain-conditioned parameters (per-sample).

    Each domain has its own weight and bias vectors, stored in embeddings.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 20) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [B, I] or [B, T, I]
        domain_id : LongTensor
            [B], domain indices.

        Returns
        -------
        Tensor
            [B, O] or [B, T, O]
        """
        B = domain_id.shape[0]
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, self.output_size)
        y = torch.matmul(x, W) + b.view(B, 1, self.output_size)
        if squeeze_T:
            y = y.squeeze(1)
        return y


class TransformerBlock(GradientCheckpointingLayer):
    """
    Standard Transformer block (pre-LN): LN → MHSA → residual, LN → MLP → residual.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, [B, T, H]

        Returns
        -------
        Tensor, [B, T, H]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------- Main Model ---------------------------------------


class SoftPromptedTransformer(nn.Module):
    """
    Multi-modal, domain-aware Transformer with optional soft prompts.

    See parameter and forward I/O descriptions inside the docstrings.
    """

    supports_gradient_checkpointing = True

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_proprio: int = 20,
        dim_time: int = 32,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        # ActionMoeDecoder
        use_action_decoder_moe: bool = False,
        decoder_expert_names: list[str] | None = None,
        router_enabled: bool = True,
        router_select_mode: str = "hard",
        decoder_routing_basis_dim: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj

        # ActionMoeDecoder
        self.use_action_decoder_moe = use_action_decoder_moe or bool(decoder_expert_names)
        self.decoder_expert_names = list(decoder_expert_names or [])
        self.router_enabled = router_enabled
        self.router_select_mode = router_select_mode
        self.decoder_routing_basis_dim = decoder_routing_basis_dim if decoder_routing_basis_dim > 0 else dim_action

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        # self.action_encoder = DomainAwareLinear(
        #     dim_action + dim_time + dim_proprio, hidden_size, num_domains=num_domains
        # )
        # self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)
        # self.embedding_to_linear = False

        self.action_encoder = nn.Linear(dim_action + dim_time + dim_proprio, hidden_size)
        self.action_decoder = nn.Linear(hidden_size, dim_action)

        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)

        self.apply(basic_init)

        if self.use_action_decoder_moe:
            logger.info(
                f"[[Model]] Initialization: {len(self.decoder_expert_names)} experts will be loaded from disk: {self.decoder_expert_names}"
            )
            self.replace_action_decoder_with_moe(
                expert_names=self.decoder_expert_names,
                router_enabled=self.router_enabled,
                router_select_mode=self.router_select_mode,
                routing_basis_dim=self.decoder_routing_basis_dim,
                is_initialization=True,
            )

    def replace_action_decoder_with_moe(
        self,
        expert_names: list[str] | None = None,
        router_enabled: bool = True,
        router_select_mode: str = "hard",
        routing_basis_dim: int | None = None,
        is_initialization: bool = False,
    ) -> ActionMoEDecoder:
        if routing_basis_dim is None or routing_basis_dim <= 0:
            routing_basis_dim = self.decoder_routing_basis_dim
        if isinstance(self.action_decoder, ActionMoEDecoder):
            base_device = self.action_decoder.base_decoder.weight.device
            base_dtype = self.action_decoder.base_decoder.weight.dtype
        elif isinstance(self.action_decoder, nn.Linear):
            base_device = self.action_decoder.weight.device
            base_dtype = self.action_decoder.weight.dtype
        else:
            raise TypeError(f"Unsupported action_decoder type: {type(self.action_decoder)!r}")
        if isinstance(self.action_decoder, ActionMoEDecoder):
            moe_decoder = self.action_decoder
        else:
            moe_decoder = ActionMoEDecoder(
                base_decoder=self.action_decoder,
                expert_names=None,
                router_enabled=router_enabled,
                router_select_mode=router_select_mode,
                routing_basis_dim=routing_basis_dim,
            )
            self.action_decoder = moe_decoder

        moe_decoder.router_enabled = router_enabled
        moe_decoder.router_select_mode = router_select_mode
        moe_decoder.routing_basis_dim = routing_basis_dim
        for expert_name in expert_names or []:
            moe_decoder.append_expert(expert_name)
            if not is_initialization:
                logger.info(f"[[Model]] Added expert '{expert_name}' to ActionMoEDecoder due to new skill training")

        moe_decoder = moe_decoder.to(device=base_device, dtype=base_dtype)
        self.action_decoder = moe_decoder

        self.use_action_decoder_moe = True
        self.decoder_expert_names = list(moe_decoder.expert_names)
        self.router_enabled = router_enabled
        self.router_select_mode = router_select_mode
        self.decoder_routing_basis_dim = routing_basis_dim
        return moe_decoder

    def set_action_decoder_active_expert(self, expert_name: str | None) -> None:
        if not isinstance(self.action_decoder, ActionMoEDecoder):
            raise RuntimeError("action_decoder is not an ActionMoEDecoder")
        self.action_decoder.set_active_expert(expert_name)

    def update_action_decoder_routing_basis(self, expert_name: str, routing_basis: torch.Tensor) -> None:
        if not isinstance(self.action_decoder, ActionMoEDecoder):
            raise RuntimeError("action_decoder is not an ActionMoEDecoder")
        self.action_decoder.update_routing_basis(expert_name, routing_basis)

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass.

        Inputs
        ------
        domain_id : [B]
        vlm_features : [B, T_vlm, D]
        aux_visual_inputs : [B, T_aux, D]
        action_with_noise : [B, T_action, dim_action]
        proprio : [B, dim_proprio]
        t : [B]

        Returns
        -------
        Tensor
            Predicted actions, [B, T_action, dim_action]
        """
        B, num_actions = action_with_noise.shape[:2]

        # Encode (action + proprio + time) → tokens
        time_emb = timestep_embedding(t, self.dim_time)  # [B, dim_time]
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        # if self.embedding_to_linear:
        #     x = self.action_encoder(action_tokens)  # [B, T_action, H]
        # else:
        #     x = self.action_encoder(action_tokens, domain_id)  # [B, T_action, H]
        x = self.action_encoder(action_tokens)  # [B, T_action, H]

        # Project visual streams and concatenate
        if self.use_hetero_proj:
            x = torch.cat(
                [x, self.vlm_proj(vlm_features, domain_id), self.aux_visual_proj(aux_visual_inputs, domain_id)],
                dim=1,
            )
        else:
            x = torch.cat([x, self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual_inputs)], dim=1)

        # Add positional embeddings (truncate if needed)
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}.")
        x = x + self.pos_emb[:, :seq_len, :]

        # Append soft prompts
        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(B, self.len_soft_prompts, self.hidden_size)
            x = torch.cat([x, soft_prompts], dim=1)

        # Transformer backbone
        for block in self.blocks:
            x = block(x)

        # Decode only the action segment
        # if self.embedding_to_linear:
        #     return self.action_decoder(self.norm(x[:, :num_actions]))
        # else:
        #     return self.action_decoder(self.norm(x[:, :num_actions]), domain_id)
        return self.action_decoder(self.norm(x[:, :num_actions]))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        is_gradient_checkpointing_set = False

        # # Apply it on the top-level module in case the top-level modules supports it
        # # for example, LongT5Stack inherits from `PreTrainedModel`.
        # if hasattr(self, "gradient_checkpointing"):
        #     print(f"{self.__class__.__name__} enabled gradient_checkpointing: {enable} ")
        #     self._gradient_checkpointing_func = gradient_checkpointing_func
        #     self.gradient_checkpointing = enable
        #     is_gradient_checkpointing_set = True

        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            self._set_gradient_checkpointing(enable=False)
