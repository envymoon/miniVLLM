from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from ..worker.batch import GPUInputBatch


@dataclass(slots=True)
class AttentionMetadata:
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    request_indices: torch.Tensor
    is_decode: torch.Tensor
    active_mask: torch.Tensor
    block_size: int

    @classmethod
    def from_batch(
        cls,
        batch: GPUInputBatch,
        device: torch.device,
        block_size: int,
        max_blocks: int | None = None,
    ) -> AttentionMetadata:
        table_width = max_blocks or len(batch.block_tables[0])
        if table_width < len(batch.block_tables[0]):
            raise ValueError("block table exceeds configured metadata width")
        padded_tables = [
            table + [-1] * (table_width - len(table))
            for table in batch.block_tables
        ]
        request_indices = [
            request
            for request in range(batch.num_requests)
            for _ in range(
                batch.query_start_loc[request + 1]
                - batch.query_start_loc[request]
            )
        ]
        return cls(
            positions=torch.tensor(batch.positions, device=device, dtype=torch.int32),
            slot_mapping=torch.tensor(
                batch.slot_mapping, device=device, dtype=torch.int32
            ),
            query_start_loc=torch.tensor(
                batch.query_start_loc, device=device, dtype=torch.int32
            ),
            seq_lens=torch.tensor(batch.seq_lens, device=device, dtype=torch.int32),
            block_tables=torch.tensor(
                padded_tables, device=device, dtype=torch.int32
            ),
            request_indices=torch.tensor(
                request_indices, device=device, dtype=torch.int32
            ),
            is_decode=torch.tensor(batch.is_decode, device=device, dtype=torch.bool),
            active_mask=torch.ones(
                batch.num_tokens, device=device, dtype=torch.bool
            ),
            block_size=block_size,
        )


class AttentionBackend(Protocol):
    graph_safe: bool

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor: ...


class TorchAttentionBackend:
    """Tensor reference for the same paged metadata consumed by CUDA kernels."""

    graph_safe = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        active_slots = metadata.slot_mapping[metadata.active_mask].long()
        key_cache.index_copy_(0, active_slots, key[metadata.active_mask])
        value_cache.index_copy_(0, active_slots, value[metadata.active_mask])

        outputs = torch.zeros_like(query)
        query_heads = query.shape[1]
        kv_heads = key.shape[1]
        if query_heads % kv_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        groups = query_heads // kv_heads
        for request in range(metadata.seq_lens.numel()):
            query_start = int(metadata.query_start_loc[request].item())
            query_end = int(metadata.query_start_loc[request + 1].item())
            if query_start == query_end:
                continue
            request_active = metadata.active_mask[query_start:query_end]
            if not bool(request_active.any().item()):
                continue
            seq_len = int(metadata.seq_lens[request].item())
            block_table = metadata.block_tables[request]
            positions = torch.arange(seq_len, device=query.device)
            slots = (
                block_table[positions // metadata.block_size].long()
                * metadata.block_size
                + positions % metadata.block_size
            )
            request_key = key_cache[slots]
            request_value = value_cache[slots]
            if groups > 1:
                request_key = request_key.repeat_interleave(groups, dim=1)
                request_value = request_value.repeat_interleave(groups, dim=1)
            request_query = query[query_start:query_end].transpose(0, 1).unsqueeze(0)
            request_key = request_key.transpose(0, 1).unsqueeze(0)
            request_value = request_value.transpose(0, 1).unsqueeze(0)
            query_positions = metadata.positions[query_start:query_end]
            causal_mask = positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            result = F.scaled_dot_product_attention(
                request_query,
                request_key,
                request_value,
                attn_mask=causal_mask.unsqueeze(0).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
            )
            outputs[query_start:query_end] = result.squeeze(0).transpose(0, 1)
        return outputs


class CudaAttentionBackend:
    """Graph-safe tensor ABI for custom flash-prefill and paged-decode kernels."""

    graph_safe = True

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("custom attention backend requires CUDA PyTorch")
        self._ops = self._load_extension()

    @staticmethod
    def _load_extension() -> object:
        try:
            from torch.utils.cpp_extension import load
        except ImportError as error:
            raise RuntimeError("PyTorch C++ extension support is unavailable") from error
        root = Path(__file__).resolve().parents[2]
        return load(
            name="minivllm_llama_attention",
            sources=[
                str(root / "runtime" / "torch_attention.cpp"),
                str(root / "runtime" / "torch_attention_cuda.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        output = torch.zeros_like(query)
        self._ops.append_kv_(
            key,
            value,
            key_cache,
            value_cache,
            metadata.slot_mapping,
            metadata.active_mask,
        )
        self._ops.flash_prefill_out(
            query,
            key_cache,
            value_cache,
            metadata.positions,
            metadata.request_indices,
            metadata.block_tables,
            metadata.is_decode,
            metadata.active_mask,
            metadata.block_size,
            output,
        )
        self._ops.paged_decode_out(
            query,
            key_cache,
            value_cache,
            metadata.query_start_loc,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.is_decode,
            metadata.active_mask,
            metadata.block_size,
            output,
        )
        return output


def create_attention_backend(name: str, device: torch.device) -> AttentionBackend:
    if name == "torch":
        return TorchAttentionBackend()
    if name == "custom":
        return CudaAttentionBackend()
    if name == "auto":
        return (
            CudaAttentionBackend()
            if device.type == "cuda"
            else TorchAttentionBackend()
        )
    raise ValueError(f"unsupported attention backend: {name}")
