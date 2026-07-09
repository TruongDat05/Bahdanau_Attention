from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]


@dataclass(frozen=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(cls, sentences: Iterable[list[str]]) -> "Vocab":
        words = sorted({word for sent in sentences for word in sent})
        itos = SPECIAL_TOKENS + words
        stoi = {word: idx for idx, word in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Vocab":
        itos = list(payload["itos"])  # type: ignore[arg-type]
        return cls(stoi={word: idx for idx, word in enumerate(itos)}, itos=itos)

    def to_dict(self) -> dict[str, list[str]]:
        return {"itos": self.itos}

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(tok, self.stoi[UNK]) for tok in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[idx] for idx in ids]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def sos_id(self) -> int:
        return self.stoi[SOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]


def strip_special(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token not in SPECIAL_TOKENS]
