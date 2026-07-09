from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset

from vocab import Vocab


def make_toy_pairs(n: int = 1000, max_len: int = 8, seed: int = 42) -> list[tuple[list[str], list[str]]]:
    """Create a tiny English -> Vietnamese parallel corpus.

    The data is synthetic but behaves like a small translation task: one source
    sentence maps to one target sentence, word order is mostly preserved, and a
    few target words expand into two tokens.
    """
    subjects = [
        (["i"], ["toi"]),
        (["you"], ["ban"]),
        (["we"], ["chung", "toi"]),
        (["they"], ["ho"]),
        (["he"], ["anh", "ay"]),
        (["she"], ["co", "ay"]),
    ]
    verbs = [
        (["like"], ["thich"]),
        (["love"], ["yeu"]),
        (["see"], ["thay"]),
        (["eat"], ["an"]),
        (["read"], ["doc"]),
        (["want"], ["muon"]),
        (["need"], ["can"]),
    ]
    objects = [
        (["apples"], ["tao"]),
        (["bananas"], ["chuoi"]),
        (["rice"], ["com"]),
        (["books"], ["sach"]),
        (["cats"], ["meo"]),
        (["dogs"], ["cho"]),
        (["water"], ["nuoc"]),
        (["coffee"], ["ca", "phe"]),
        (["music"], ["nhac"]),
    ]
    suffixes = [
        ([], []),
        (["today"], ["hom", "nay"]),
        (["tomorrow"], ["ngay", "mai"]),
        (["now"], ["bay", "gio"]),
        (["at", "home"], ["o", "nha"]),
        (["in", "school"], ["o", "truong"]),
        (["every", "day"], ["moi", "ngay"]),
    ]

    all_pairs: list[tuple[list[str], list[str]]] = []
    for src_subj, tgt_subj in subjects:
        for src_verb, tgt_verb in verbs:
            for src_obj, tgt_obj in objects:
                for src_suffix, tgt_suffix in suffixes:
                    src = src_subj + src_verb + src_obj + src_suffix
                    tgt = tgt_subj + tgt_verb + tgt_obj + tgt_suffix
                    if len(src) <= max_len and len(tgt) <= max_len:
                        all_pairs.append((src, tgt))

    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    return all_pairs[:n]


def pad_to_len(ids: list[int], length: int, pad_id: int) -> list[int]:
    return ids + [pad_id] * (length - len(ids))


class TranslationDataset(Dataset):
    """Return tensors used by teacher forcing.

    src:     [source words..., EOS, PAD...]
    tgt_in:  [SOS, target words..., PAD...]
    tgt_out: [target words..., EOS, PAD...]
    """

    def __init__(
        self,
        pairs: list[tuple[list[str], list[str]]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int,
    ) -> None:
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_seq_len = max_len + 1
        self.tgt_seq_len = max_len + 1

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_tokens, tgt_tokens = self.pairs[idx]

        src_ids = self.src_vocab.encode(src_tokens) + [self.src_vocab.eos_id]
        tgt_in_ids = [self.tgt_vocab.sos_id] + self.tgt_vocab.encode(tgt_tokens)
        tgt_out_ids = self.tgt_vocab.encode(tgt_tokens) + [self.tgt_vocab.eos_id]

        src = pad_to_len(src_ids, self.src_seq_len, self.src_vocab.pad_id)
        tgt_in = pad_to_len(tgt_in_ids, self.tgt_seq_len, self.tgt_vocab.pad_id)
        tgt_out = pad_to_len(tgt_out_ids, self.tgt_seq_len, self.tgt_vocab.pad_id)

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_in, dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
        )


def build_splits(
    num_pairs: int,
    max_len: int,
    seed: int,
    train_ratio: float,
) -> tuple[list[tuple[list[str], list[str]]], list[tuple[list[str], list[str]]], Vocab, Vocab]:
    pairs = make_toy_pairs(num_pairs, max_len, seed)
    split = int(train_ratio * len(pairs))
    train_pairs = pairs[:split]
    valid_pairs = pairs[split:]
    src_vocab = Vocab.build(src for src, _ in pairs)
    tgt_vocab = Vocab.build(tgt for _, tgt in pairs)
    return train_pairs, valid_pairs, src_vocab, tgt_vocab


def build_loaders(
    train_pairs: list[tuple[list[str], list[str]]],
    valid_pairs: list[tuple[list[str], list[str]]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_len: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab, max_len)
    valid_ds = TranslationDataset(valid_pairs, src_vocab, tgt_vocab, max_len)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    return train_loader, valid_loader
