from __future__ import annotations

from .operator_runtime import KernelRuntime
from ..protocol import WorkerItem
from ..tokenizer import CharacterTokenizer


class BootstrapModelRunner:
    """Deterministic bootstrap runner used until model layers are connected."""

    _response = " mini-vLLM"

    def __init__(self, runtime: KernelRuntime, block_size: int) -> None:
        self.runtime = runtime
        self.block_size = block_size
        self.tokenizer = CharacterTokenizer()
        self._response_tokens = self.tokenizer.encode(self._response)

    def execute(self, item: WorkerItem) -> int | None:
        self.runtime.append_kv(item.token_ids, item.slot_mapping)
        if item.generated_count == 0:
            prefix_slots = [
                item.block_table[position // self.block_size] * self.block_size
                + position % self.block_size
                for position in range(item.seq_len)
            ]
            self.runtime.flash_attention_prefill(prefix_slots)
        else:
            self.runtime.paged_decode_attention(
                item.block_table, item.seq_len, self.block_size
            )
        if not item.should_sample:
            return None
        if item.generated_count >= len(self._response_tokens):
            return self.tokenizer.eos_token_id
        return self._response_tokens[item.generated_count]
