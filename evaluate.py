from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import load_config
from dataset import build_loaders, build_splits
from model import Seq2Seq
from translate import choose_device, load_checkpoint, translate_sentence


@torch.no_grad()
def evaluate_loss(
    model: Seq2Seq,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_id: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for src, tgt_in, tgt_out in loader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()

        pred = logits.argmax(dim=-1)
        mask = tgt_out.ne(pad_id)
        correct += pred.eq(tgt_out).logical_and(mask).sum().item()
        total += mask.sum().item()

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def corpus_bleu(references: Iterable[list[str]], hypotheses: Iterable[list[str]], max_n: int = 4) -> float:
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    hyp_len = 0

    for reference, hypothesis in zip(references, hypotheses):
        ref_len += len(reference)
        hyp_len += len(hypothesis)
        for n in range(1, max_n + 1):
            ref_ngrams = ngram_counts(reference, n)
            hyp_ngrams = ngram_counts(hypothesis, n)
            total_counts[n - 1] += sum(hyp_ngrams.values())
            clipped_counts[n - 1] += sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())

    if hyp_len == 0:
        return 0.0

    precisions = []
    for clipped, total in zip(clipped_counts, total_counts):
        if total == 0:
            precisions.append(1.0)
        elif clipped == 0:
            precisions.append(1.0 / (2.0 * total))
        else:
            precisions.append(clipped / total)

    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))
    return 100.0 * brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))


@torch.no_grad()
def evaluate_bleu(
    model: Seq2Seq,
    pairs: list[tuple[list[str], list[str]]],
    src_vocab,
    tgt_vocab,
    config: dict,
    device: torch.device,
) -> tuple[float, list[tuple[str, str, str]]]:
    references: list[list[str]] = []
    hypotheses: list[list[str]] = []
    examples: list[tuple[str, str, str]] = []
    decoding = config["decoding"]

    for src_tokens, tgt_tokens in pairs:
        source = " ".join(src_tokens)
        result = translate_sentence(
            model,
            source,
            src_vocab,
            tgt_vocab,
            max_len=int(config["data"]["max_len"]),
            device=device,
            beam_size=int(decoding["beam_size"]),
            length_penalty=float(decoding["length_penalty"]),
            max_extra_len=int(decoding["max_extra_len"]),
        )
        references.append(tgt_tokens)
        hypotheses.append(result.tokens)
        if len(examples) < 5:
            examples.append((source, " ".join(tgt_tokens), result.text))

    return corpus_bleu(references, hypotheses), examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation loss, token accuracy, and BLEU.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device("cpu" if args.cpu else str(config["device"]))
    checkpoint_path = args.checkpoint or config["paths"]["best_checkpoint"]
    model, src_vocab, tgt_vocab, checkpoint_config = load_checkpoint(checkpoint_path, device)
    config = checkpoint_config

    train_pairs, valid_pairs, _, _ = build_splits(
        num_pairs=int(config["data"]["num_pairs"]),
        max_len=int(config["data"]["max_len"]),
        seed=int(config["seed"]),
        train_ratio=float(config["data"]["train_ratio"]),
    )
    _, valid_loader = build_loaders(
        train_pairs,
        valid_pairs,
        src_vocab,
        tgt_vocab,
        max_len=int(config["data"]["max_len"]),
        batch_size=int(config["train"]["batch_size"]),
        seed=int(config["seed"]),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
    valid_loss, token_acc = evaluate_loss(model, valid_loader, criterion, device, tgt_vocab.pad_id)
    bleu, examples = evaluate_bleu(model, valid_pairs, src_vocab, tgt_vocab, config, device)

    print(f"valid loss: {valid_loss:.4f}")
    print(f"valid token acc: {token_acc:.3f}")
    print(f"valid BLEU: {bleu:.2f}")
    print("\nExamples")
    for source, reference, hypothesis in examples:
        print(f"{source:30s} ref: {reference:28s} hyp: {hypothesis}")


if __name__ == "__main__":
    main()
