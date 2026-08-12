from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attention import (
    AttentionBackend,
    AttentionMetadata,
    create_attention_backend,
)
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
    def __init__(self, config: LlamaConfig, backend: AttentionBackend) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
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
        metadata: AttentionMetadata,
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

        attention = self.backend.forward(
            query, key, value, key_cache, value_cache, metadata
        )
        return self.o_proj(attention.reshape(-1, self.config.hidden_size))


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
    def __init__(self, config: LlamaConfig, backend: AttentionBackend) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, backend)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            positions,
            metadata,
            key_cache,
            value_cache,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig, backend: AttentionBackend) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(config, backend)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class LlamaForCausalLM(nn.Module):
    def __init__(
        self, config: LlamaConfig, backend: AttentionBackend | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend or create_attention_backend("torch", torch.device("cpu"))
        self.model = LlamaModel(config, self.backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        key_cache: list[torch.Tensor],
        value_cache: list[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.model.embed_tokens(input_ids)
        for index, layer in enumerate(self.model.layers):
            hidden_states = layer(
                hidden_states,
                positions,
                metadata,
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


@dataclass(slots=True)
class LlamaCUDAGraphState:
    input_ids: torch.Tensor
    positions: torch.Tensor
    metadata: AttentionMetadata
    graph: torch.cuda.CUDAGraph | None = None
    logits: torch.Tensor | None = None


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
        max_num_seqs: int = 32,
        attention_backend: str = "auto",
    ) -> None:
        self.config = LlamaConfig.from_model_dir(model_path)
        if max_model_len > self.config.max_position_embeddings:
            raise ValueError(
                "max_model_len exceeds the model's max_position_embeddings"
            )
        self.device = resolve_torch_device(device, data_parallel_rank)
        self.dtype = resolve_torch_dtype(dtype, self.device)
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_blocks = (max_model_len + block_size - 1) // block_size
        self.attention_backend = create_attention_backend(
            attention_backend, self.device
        )
        self.model = LlamaForCausalLM(self.config, self.attention_backend)
        load_safetensor_weights(self.model, model_path)
        self.model.to(device=self.device, dtype=self.dtype).eval()

        self.physical_cache_slots = num_gpu_blocks * block_size
        cache_shape = (
            self.physical_cache_slots + max_num_seqs,
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
        self._graph_states: dict[int, LlamaCUDAGraphState] = {}

    @property
    def supports_cudagraph(self) -> bool:
        return self.device.type == "cuda" and self.attention_backend.graph_safe

    def _metadata(self, batch: GPUInputBatch) -> AttentionMetadata:
        return AttentionMetadata.from_batch(
            batch,
            self.device,
            self.block_size,
            max_blocks=self.max_blocks,
        )

    def _sample_logits(
        self, batch: GPUInputBatch, logits: torch.Tensor
    ) -> dict[str, int]:
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

    @torch.inference_mode()
    def execute_model(self, batch: GPUInputBatch) -> dict[str, int]:
        input_ids = torch.tensor(batch.input_ids, device=self.device, dtype=torch.long)
        positions = torch.tensor(batch.positions, device=self.device, dtype=torch.long)
        metadata = self._metadata(batch)
        logits = self.model(
            input_ids, positions, metadata, self.key_cache, self.value_cache
        )
        return self._sample_logits(batch, logits)

    def _create_graph_state(self, capture_size: int) -> LlamaCUDAGraphState:
        if capture_size > self.max_num_seqs:
            raise ValueError("capture size exceeds max_num_seqs")
        input_ids = torch.zeros(capture_size, device=self.device, dtype=torch.long)
        positions = torch.zeros(capture_size, device=self.device, dtype=torch.long)
        metadata = AttentionMetadata(
            positions=torch.zeros(
                capture_size, device=self.device, dtype=torch.int32
            ),
            slot_mapping=torch.empty(
                capture_size, device=self.device, dtype=torch.int32
            ),
            query_start_loc=torch.arange(
                capture_size + 1, device=self.device, dtype=torch.int32
            ),
            seq_lens=torch.ones(
                capture_size, device=self.device, dtype=torch.int32
            ),
            block_tables=torch.zeros(
                (capture_size, self.max_blocks),
                device=self.device,
                dtype=torch.int32,
            ),
            request_indices=torch.arange(
                capture_size, device=self.device, dtype=torch.int32
            ),
            is_decode=torch.ones(
                capture_size, device=self.device, dtype=torch.bool
            ),
            active_mask=torch.zeros(
                capture_size, device=self.device, dtype=torch.bool
            ),
            block_size=self.block_size,
        )
        return LlamaCUDAGraphState(input_ids, positions, metadata)

    def _copy_decode_batch(
        self, state: LlamaCUDAGraphState, batch: GPUInputBatch
    ) -> None:
        if not batch.descriptor.uniform_decode:
            raise ValueError("Llama CUDA Graph requires a uniform decode batch")
        count = batch.num_requests
        capture_size = state.input_ids.numel()
        if count > capture_size:
            raise ValueError("batch exceeds CUDA Graph capture size")
        state.input_ids.zero_()
        state.positions.zero_()
        state.metadata.positions.zero_()
        state.metadata.seq_lens.fill_(1)
        state.metadata.block_tables.zero_()
        state.metadata.active_mask.zero_()
        state.input_ids[:count].copy_(
            torch.tensor(batch.input_ids, device=self.device, dtype=torch.long)
        )
        state.positions[:count].copy_(
            torch.tensor(batch.positions, device=self.device, dtype=torch.long)
        )
        state.metadata.positions[:count].copy_(
            torch.tensor(batch.positions, device=self.device, dtype=torch.int32)
        )
        state.metadata.slot_mapping[:count].copy_(
            torch.tensor(batch.slot_mapping, device=self.device, dtype=torch.int32)
        )
        state.metadata.seq_lens[:count].copy_(
            torch.tensor(batch.seq_lens, device=self.device, dtype=torch.int32)
        )
        table_width = len(batch.block_tables[0])
        state.metadata.block_tables[:count, :table_width].copy_(
            torch.tensor(
                batch.block_tables, device=self.device, dtype=torch.int32
            )
        )
        state.metadata.active_mask[:count] = True
        if count < capture_size:
            state.metadata.slot_mapping[count:].copy_(
                torch.arange(
                    self.physical_cache_slots + count,
                    self.physical_cache_slots + capture_size,
                    device=self.device,
                    dtype=torch.int32,
                )
            )

    def _capture_graph(self, state: LlamaCUDAGraphState) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.model(
                    state.input_ids,
                    state.positions,
                    state.metadata,
                    self.key_cache,
                    self.value_cache,
                )
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state.logits = self.model(
                state.input_ids,
                state.positions,
                state.metadata,
                self.key_cache,
                self.value_cache,
            )
        state.graph = graph

    @torch.inference_mode()
    def execute_cudagraph(
        self, batch: GPUInputBatch, capture_size: int
    ) -> dict[str, int]:
        if not self.supports_cudagraph:
            return self.execute_model(batch)
        state = self._graph_states.get(capture_size)
        if state is None:
            state = self._create_graph_state(capture_size)
            self._graph_states[capture_size] = state
        self._copy_decode_batch(state, batch)
        if state.graph is None:
            self._capture_graph(state)
        else:
            state.graph.replay()
        if state.logits is None:
            raise RuntimeError("CUDA Graph did not produce logits")
        return self._sample_logits(batch, state.logits)

    def close(self) -> None:
        self.key_cache.clear()
        self.value_cache.clear()
        self._graph_states.clear()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
