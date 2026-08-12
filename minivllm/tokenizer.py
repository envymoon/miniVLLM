from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Tokenizer(Protocol):
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str: ...


class CharacterTokenizer:
    """Dependency-free tokenizer used by the bootstrap model runner.

    Token ids 0..2 are control tokens. Unicode code points are shifted by three,
    making encode/decode lossless without a vocabulary file.
    """

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    _offset = 3

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("prompt must be a string")
        return [ord(char) + self._offset for char in text]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return "".join(
            chr(token_id - self._offset)
            for token_id in token_ids
            if token_id >= self._offset
        )


class HuggingFaceTokenizer:
    """Local-only tokenizer adapter used when a model directory is configured."""

    def __init__(self, path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "model tokenization requires the optional 'model' dependencies"
            ) from error
        if not Path(path).exists():
            raise FileNotFoundError(f"tokenizer path does not exist: {path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, use_fast=True
        )
        eos = self._tokenizer.eos_token_id
        bos = self._tokenizer.bos_token_id
        pad = self._tokenizer.pad_token_id
        if eos is None:
            raise ValueError("tokenizer must define eos_token_id")
        self.eos_token_id = int(eos)
        self.bos_token_id = int(bos if bos is not None else eos)
        self.pad_token_id = int(pad if pad is not None else eos)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("prompt must be a string")
        return list(self._tokenizer.encode(text, add_special_tokens=True))

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return self._tokenizer.decode(
            list(token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


def create_tokenizer(
    model_path: str | None = None, tokenizer_path: str | None = None
) -> Tokenizer:
    path = tokenizer_path or model_path
    if path is None:
        return CharacterTokenizer()
    return HuggingFaceTokenizer(path)
