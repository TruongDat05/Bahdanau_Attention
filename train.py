from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import load_config
from dataset import build_loaders, build_splits
from evaluate import evaluate_bleu, evaluate_loss
from model import Seq2Seq, build_model
from translate import choose_device, save_attention_heatmap, translate_sentence


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: Seq2Seq,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip: float,
) -> float:
    model.train()
    total_loss = 0.0

    for src, tgt_in, tgt_out in loader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def trace_tensor_shapes(model: Seq2Seq, batch, device: torch.device) -> None:
    src, tgt_in, tgt_out = [x.to(device) for x in batch]
    encoder = model.encoder
    decoder = model.decoder

    with torch.no_grad():
        src_emb = encoder.embedding(src)
        enc_outputs, raw_enc_hidden = encoder.gru(src_emb)
        enc_hidden_cat = torch.cat([raw_enc_hidden[-2], raw_enc_hidden[-1]], dim=-1)
        decoder_init = torch.tanh(encoder.hidden_bridge(enc_hidden_cat)).unsqueeze(0)
        tgt_emb = decoder.embedding(tgt_in)
        src_mask = src.ne(model.src_pad_id)
        dec_output = decoder(tgt_in, enc_outputs, decoder_init, src_mask)
        one_step_rnn_input_dim = tgt_emb.size(-1) + dec_output.context_vectors.size(-1)

    print("\nTensor shape trace for one batch")
    print(f"src ids                {tuple(src.shape)}      = (batch, src_seq_len)")
    print(f"src embedding          {tuple(src_emb.shape)}  = (batch, src_seq_len, emb_dim)")
    print(f"Bi-GRU outputs h_1..T  {tuple(enc_outputs.shape)} = (batch, src_seq_len, hidden_dim * 2)")
    print(f"raw Bi-GRU hidden      {tuple(raw_enc_hidden.shape)} = (2, batch, hidden_dim)")
    print(f"decoder init s_0       {tuple(decoder_init.shape)} = (1, batch, hidden_dim)")
    print(f"tgt_in ids             {tuple(tgt_in.shape)}      = (batch, tgt_seq_len)")
    print(f"tgt embedding          {tuple(tgt_emb.shape)}  = (batch, tgt_seq_len, emb_dim)")
    print(f"attention weights a_tj {tuple(dec_output.attention_weights.shape)} = (batch, tgt_seq_len, src_seq_len)")
    print(f"context vectors c_t    {tuple(dec_output.context_vectors.shape)} = sum_j a_tj * h_j")
    print(f"decoder GRU input step (batch, 1, {one_step_rnn_input_dim}) = emb_dim + hidden_dim * 2")
    print(f"decoder outputs s_1..T {tuple(dec_output.decoder_outputs.shape)} = (batch, tgt_seq_len, hidden_dim)")
    print(f"logits                 {tuple(dec_output.logits.shape)} = (batch, tgt_seq_len, tgt_vocab_size)")
    print(f"tgt_out labels         {tuple(tgt_out.shape)}      = (batch, tgt_seq_len)\n")


def save_checkpoint(
    path: str | Path,
    model: Seq2Seq,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    src_vocab,
    tgt_vocab,
    metrics: dict[str, float],
    history: list[dict[str, float]],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "src_vocab": src_vocab.to_dict(),
            "tgt_vocab": tgt_vocab.to_dict(),
            "metrics": metrics,
            "history": history,
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_loss_curve(history: list[dict[str, float]], path: str | Path) -> Path:
    svg_path = Path(path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        svg_path.write_text("", encoding="utf-8")
        return svg_path

    width = 760
    height = 420
    left = 58
    right = 24
    top = 28
    bottom = 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    valid = [row["valid_loss"] for row in history]
    y_values = train + valid
    y_min = min(y_values)
    y_max = max(y_values)
    if y_min == y_max:
        y_max = y_min + 1.0

    def xy(epoch: float, value: float) -> tuple[float, float]:
        x = left + (epoch - min(epochs)) / max(max(epochs) - min(epochs), 1) * plot_w
        y = top + (y_max - value) / (y_max - y_min) * plot_h
        return x, y

    def points(values: list[float]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(epoch, value) for epoch, value in zip(epochs, values)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="24" font-family="Arial, sans-serif" font-size="18" font-weight="700">Training loss curve</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{points(train)}"/>',
        f'<polyline fill="none" stroke="#dc2626" stroke-width="3" points="{points(valid)}"/>',
        f'<text x="{left}" y="{height - 16}" font-family="Arial, sans-serif" font-size="12">epoch {min(epochs):.0f}</text>',
        f'<text x="{left + plot_w - 52}" y="{height - 16}" font-family="Arial, sans-serif" font-size="12">epoch {max(epochs):.0f}</text>',
        f'<text x="8" y="{top + 8}" font-family="Arial, sans-serif" font-size="12">{y_max:.2f}</text>',
        f'<text x="8" y="{top + plot_h}" font-family="Arial, sans-serif" font-size="12">{y_min:.2f}</text>',
        '<rect x="570" y="32" width="14" height="14" fill="#2563eb"/>',
        '<text x="590" y="44" font-family="Arial, sans-serif" font-size="13">train loss</text>',
        '<rect x="570" y="54" width="14" height="14" fill="#dc2626"/>',
        '<text x="590" y="66" font-family="Arial, sans-serif" font-size="13">valid loss</text>',
    ]
    for epoch, value in zip(epochs, train):
        x, y = xy(epoch, value)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/>')
    for epoch, value in zip(epochs, valid):
        x, y = xy(epoch, value)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#dc2626"/>')
    parts.append("</svg>")
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return svg_path


def write_history(history: list[dict[str, float]], checkpoint_dir: str | Path) -> Path:
    path = Path(checkpoint_dir) / "history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Level 2 Bahdanau attention model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.cpu:
        config["device"] = "cpu"

    set_seed(int(config["seed"]))
    device = choose_device(str(config["device"]))

    train_pairs, valid_pairs, src_vocab, tgt_vocab = build_splits(
        num_pairs=int(config["data"]["num_pairs"]),
        max_len=int(config["data"]["max_len"]),
        seed=int(config["seed"]),
        train_ratio=float(config["data"]["train_ratio"]),
    )
    train_loader, valid_loader = build_loaders(
        train_pairs,
        valid_pairs,
        src_vocab,
        tgt_vocab,
        max_len=int(config["data"]["max_len"]),
        batch_size=int(config["train"]["batch_size"]),
        seed=int(config["seed"]),
    )
    model = build_model(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        emb_dim=int(config["model"]["emb_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))

    print(f"device: {device}")
    print(f"pairs: {len(train_pairs) + len(valid_pairs)} train: {len(train_pairs)} valid: {len(valid_pairs)}")
    print(f"src vocab: {len(src_vocab.itos)} tgt vocab: {len(tgt_vocab.itos)}")
    if not args.no_trace:
        trace_tensor_shapes(model, next(iter(train_loader)), device)

    history: list[dict[str, float]] = []
    best_bleu = -1.0
    best_path = Path(config["paths"]["best_checkpoint"])
    latest_path = Path(config["paths"]["latest_checkpoint"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            clip=float(config["train"]["clip"]),
        )
        valid_loss, valid_acc = evaluate_loss(model, valid_loader, criterion, device, tgt_vocab.pad_id)
        bleu, _ = evaluate_bleu(model, valid_pairs, src_vocab, tgt_vocab, config, device)
        metrics = {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_token_acc": valid_acc,
            "valid_bleu": bleu,
        }
        history.append({"epoch": float(epoch), **metrics})

        save_checkpoint(latest_path, model, optimizer, epoch, config, src_vocab, tgt_vocab, metrics, history)
        save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, config, src_vocab, tgt_vocab, metrics, history)
        if bleu >= best_bleu:
            best_bleu = bleu
            save_checkpoint(best_path, model, optimizer, epoch, config, src_vocab, tgt_vocab, metrics, history)

        print(
            f"epoch {epoch:02d} | train loss {train_loss:.4f} | valid loss {valid_loss:.4f} | "
            f"valid token acc {valid_acc:.3f} | BLEU {bleu:.2f}"
        )

    history_path = write_history(history, checkpoint_dir)
    curve_path = save_loss_curve(history, config["paths"]["loss_curve"])

    print("\nExample translations")
    for sentence in config["examples"]:
        result = translate_sentence(
            model,
            str(sentence),
            src_vocab,
            tgt_vocab,
            max_len=int(config["data"]["max_len"]),
            device=device,
            beam_size=int(config["decoding"]["beam_size"]),
            length_penalty=float(config["decoding"]["length_penalty"]),
            max_extra_len=int(config["decoding"]["max_extra_len"]),
        )
        print(f"{str(sentence):30s} -> {result.text}")

    first_example = str(config["examples"][0])
    result = translate_sentence(
        model,
        first_example,
        src_vocab,
        tgt_vocab,
        max_len=int(config["data"]["max_len"]),
        device=device,
        beam_size=int(config["decoding"]["beam_size"]),
        length_penalty=float(config["decoding"]["length_penalty"]),
        max_extra_len=int(config["decoding"]["max_extra_len"]),
    )
    heatmap_path = save_attention_heatmap(
        result.attention,
        result.src_tokens,
        result.attention_tokens,
        config["paths"]["attention_heatmap"],
    )
    print(f"\ncheckpoints: {checkpoint_dir}")
    print(f"best checkpoint: {best_path}")
    print(f"history: {history_path}")
    print(f"loss curve: {curve_path}")
    print(f"attention heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
