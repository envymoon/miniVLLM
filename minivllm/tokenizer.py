from __future__ import annotations


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
