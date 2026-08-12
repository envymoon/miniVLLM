from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..worker.batch import GPUInputBatch


@dataclass(frozen=True, slots=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_model_dir(cls, model_path: str | Path) -> LlamaConfig:
        path = Path(model_path) / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing model config: {path}")
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        model_type = raw.get("model_type")
        if model_type not in {"llama", None}:
            raise ValueError(f"unsupported model_type: {model_type}")
        if raw.get("rope_scaling") is not None and raw.get("rope_scaling") is not False:
            raise ValueError("rope_scaling is not supported by the minimal Llama runner")
        heads = int(raw["num_attention_heads"])
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=int(raw.get("num_key_value_heads", heads)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 2048)),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            attention_bias=bool(raw.get("attention_bias", False)),
            mlp_bias=bool(raw.get("mlp_bias", False)),
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    if head_dim % 2:
        raise ValueError("rotary head dimension must be even")
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=query.device, dtype=torch.float32)
            / head_dim
        )
    )
    frequencies = torch.outer(positions.float(), inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().to(query.dtype).unsqueeze(1)
    sin = embedding.sin().to(query.dtype).unsqueeze(1)
    return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch: GPUInputBatch,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        head_dim = self.config.head_dim
        query = self.q_proj(hidden_states).view(
            -1, self.config.num_attention_heads, head_dim
        )
        key = self.k_proj(hidden_states).view(
            -1, self.config.num_key_value_heads, head_dim
        )
        value = self.v_proj(hidden_states).view(
            -1, self.config.num_key_value_heads, head_dim
        )
        query, key = apply_rotary_embedding(
            query, key, positions, self.config.rope_theta
        )

        slots = torch.tensor(batch.slot_mapping, device=hidden_states.device)
        key_cache[slots] = key
        value_cache[slots] = value

        outputs: list[torch.Tensor] = []
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        if self.config.num_attention_heads % self.config.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        for request_index in range(batch.num_requests):
            query_start = batch.query_start_loc[request_index]
            query_end = batch.query_start_loc[request_index + 1]
            context_start = batch.context_start_loc[request_index]
            context_end = batch.context_start_loc[request_index + 1]
            context_slots = torch.tensor(
                batch.context_slot_mapping[context_start:context_end],
                device=hidden_states.device,
            )
            request_key = key_cache[context_slots]
            request_value = value_cache[context_slots]
            if groups > 1:
                request_key = request_key.repeat_interleave(groups, dim=1)
                request_value = request_value.repeat_interleave(groups, dim=1)

            request_query = query[query_start:query_end].transpose(0, 1).unsqueeze(0)
            request_key = request_key.transpose(0, 1).unsqueeze(0)
            request_value = request_value.transpose(0, 1).unsqueeze(0)
            query_positions = positions[query_start:query_end]
            key_positions = torch.arange(
                batch.seq_lens[request_index], device=hidden_states.device
            )
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attention = F.scaled_dot_product_attention(
                request_query,
                request_key,
                request_value,
                attn_mask=causal_mask.unsqueeze(0).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
            )
            outputs.append(attention.squeeze(0).transpose(0, 1))
        return self.o_proj(torch.cat(outputs, dim=0).reshape(-1, self.config.hidden_size))


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch: GPUInputBatch,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            positions,
            batch,
            key_cache,
            value_cache,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: GPUInputBatch,
        key_cache: list[torch.Tensor],
        value_cache: list[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.model.embed_tokens(input_ids)
        for index, layer in enumerate(self.model.layers):
            hidden_states = layer(
                hidden_states,
                positions,
                batch,
                key_cache[index],
                value_cache[index],
            )
        return self.lm_head(self.model.norm(hidden_states))


def load_safetensor_weights(model: LlamaForCausalLM, model_path: str | Path) -> None:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError(
            "Llama weight loading requires the optional 'model' dependencies"
        ) from error

    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filenames = sorted(set(index["weight_map"].values()))
        weight_files = [root / filename for filename in filenames]
    else:
        weight_files = sorted(root.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no safetensors weights found in {root}")

    state_dict: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        state_dict.update(load_file(str(weight_file), device="cpu"))
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    missing = set(incompatible.missing_keys) - allowed_missing
    unexpected = {
        key for key in incompatible.unexpected_keys if not key.endswith("rotary_emb.inv_freq")
    }
    if missing or unexpected:
        raise ValueError(
            f"incompatible Llama weights; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight


def resolve_torch_device(device: str, data_parallel_rank: int) -> torch.device:
    if device == "auto":
        return torch.device(
            f"cuda:{data_parallel_rank}" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(device)


def resolve_torch_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


class LlamaModelRunner:
    """Correctness-first Llama runner with Worker-owned paged layer KV caches."""

    def __init__(
        self,
        model_path: str,
        num_gpu_blocks: int,
        block_size: int,
        max_model_len: int,
        device: str,
        dtype: str,
        data_parallel_rank: int,
    ) -> None:
        self.config = LlamaConfig.from_model_dir(model_path)
        if max_model_len > self.config.max_position_embeddings:
            raise ValueError(
                "max_model_len exceeds the model's max_position_embeddings"
            )
        self.device = resolve_torch_device(device, data_parallel_rank)
        self.dtype = resolve_torch_dtype(dtype, self.device)
        self.model = LlamaForCausalLM(self.config)
        load_safetensor_weights(self.model, model_path)
        self.model.to(device=self.device, dtype=self.dtype).eval()

        cache_shape = (
            num_gpu_blocks * block_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        self.key_cache = [
            torch.empty(cache_shape, device=self.device, dtype=self.dtype)
            for _ in range(self.config.num_hidden_layers)
        ]
        self.value_cache = [
            torch.empty(cache_shape, device=self.device, dtype=self.dtype)
            for _ in range(self.config.num_hidden_layers)
        ]

    @torch.inference_mode()
    def execute_model(self, batch: GPUInputBatch) -> dict[str, int]:
        input_ids = torch.tensor(batch.input_ids, device=self.device, dtype=torch.long)
        positions = torch.tensor(batch.positions, device=self.device, dtype=torch.long)
        logits = self.model(
            input_ids, positions, batch, self.key_cache, self.value_cache
        )
        sampled: dict[str, int] = {}
        for index, request_id in enumerate(batch.request_ids):
            sample_index = batch.sample_indices[index]
            if sample_index < 0:
                continue
            request_logits = logits[sample_index].float()
            temperature = batch.temperatures[index]
            if temperature == 0:
                token_id = int(torch.argmax(request_logits).item())
            else:
                probabilities = torch.softmax(request_logits / temperature, dim=-1)
                token_id = int(torch.multinomial(probabilities, 1).item())
            sampled[request_id] = token_id
        return sampled

    def close(self) -> None:
        self.key_cache.clear()
        self.value_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
