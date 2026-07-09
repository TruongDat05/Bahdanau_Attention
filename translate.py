from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from config import load_config
from dataset import pad_to_len
from model import Seq2Seq, build_model
from vocab import EOS, SOS, SPECIAL_TOKENS, Vocab, strip_special


@dataclass
class TranslationResult:
    text: str
    tokens: list[str]
    src_tokens: list[str]
    attention_tokens: list[str]
    attention: torch.Tensor
    score: float


def choose_device(config_device: str = "auto") -> torch.device:
    if config_device == "cpu":
        return torch.device("cpu")
    if config_device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[Seq2Seq, Vocab, Vocab, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    src_vocab = Vocab.from_dict(checkpoint["src_vocab"])
    tgt_vocab = Vocab.from_dict(checkpoint["tgt_vocab"])
    model = build_model(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        emb_dim=int(config["model"]["emb_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab, config


@torch.no_grad()
def translate_sentence(
    model: Seq2Seq,
    sentence: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_len: int,
    device: torch.device,
    beam_size: int = 3,
    length_penalty: float = 0.7,
    max_extra_len: int = 2,
) -> TranslationResult:
    model.eval()
    src_tokens = sentence.lower().split()[:max_len]
    src_labels = src_tokens + [EOS]
    src_ids = src_vocab.encode(src_tokens) + [src_vocab.eos_id]
    src_ids = pad_to_len(src_ids, max_len + 1, src_vocab.pad_id)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    src_mask = src.ne(src_vocab.pad_id)
    encoder_outputs, encoder_hidden = model.encoder(src)
    max_decode_len = max_len + max_extra_len

    beams: list[tuple[list[int], torch.Tensor, float, list[torch.Tensor], bool]] = [
        ([tgt_vocab.sos_id], encoder_hidden, 0.0, [], False)
    ]

    for _ in range(max_decode_len):
        candidates: list[tuple[list[int], torch.Tensor, float, list[torch.Tensor], bool]] = []
        for token_ids, hidden, log_prob, rows, finished in beams:
            if finished:
                candidates.append((token_ids, hidden, log_prob, rows, True))
                continue

            next_id = torch.tensor([[token_ids[-1]]], dtype=torch.long, device=device)
            decoder_output = model.decoder(next_id, encoder_outputs, encoder_hidden, src_mask, hidden)
            step_log_probs = torch.log_softmax(decoder_output.logits[:, -1, :], dim=-1).squeeze(0)
            top_log_probs, top_ids = torch.topk(step_log_probs, k=min(beam_size, step_log_probs.numel()))

            for next_log_prob, next_token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                next_rows = rows + [decoder_output.attention_weights[0, -1, : len(src_labels)].cpu()]
                is_finished = next_token_id == tgt_vocab.eos_id
                candidates.append(
                    (
                        token_ids + [int(next_token_id)],
                        decoder_output.hidden.clone(),
                        log_prob + float(next_log_prob),
                        next_rows,
                        is_finished,
                    )
                )

        beams = sorted(
            candidates,
            key=lambda item: normalized_score(item[2], len(item[0]) - 1, length_penalty),
            reverse=True,
        )[:beam_size]
        if all(item[4] for item in beams):
            break

    best_ids, _, best_log_prob, attention_rows, _ = max(
        beams,
        key=lambda item: normalized_score(item[2], len(item[0]) - 1, length_penalty),
    )
    decoded_ids = [idx for idx in best_ids[1:] if idx != tgt_vocab.eos_id]
    tokens = strip_special(tgt_vocab.decode(decoded_ids))
    attention_tokens = [tgt_vocab.itos[idx] for idx in best_ids[1:] if idx != tgt_vocab.sos_id]
    if not attention_tokens:
        attention_tokens = [EOS]
    if attention_rows:
        attention = torch.stack(attention_rows[: len(attention_tokens)], dim=0)
    else:
        attention = torch.empty(0, len(src_labels))
    return TranslationResult(
        text=" ".join(tokens),
        tokens=tokens,
        src_tokens=src_labels,
        attention_tokens=attention_tokens,
        attention=attention,
        score=normalized_score(best_log_prob, len(best_ids) - 1, length_penalty),
    )


def normalized_score(log_prob: float, length: int, length_penalty: float) -> float:
    return log_prob / max(length, 1) ** length_penalty


def save_attention_heatmap(
    attention_weights: torch.Tensor,
    src_tokens: list[str],
    tgt_tokens: list[str],
    save_path: str | Path,
) -> Path:
    """Write a dependency-free SVG attention heatmap."""
    import html

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = attention_weights.detach().cpu().tolist()
    cell = 40
    left = 132
    top = 74
    width = left + cell * max(len(src_tokens), 1) + 36
    height = top + cell * max(len(tgt_tokens), 1) + 46

    def color(value: float) -> str:
        value = max(0.0, min(1.0, float(value)))
        red = int(250 - 210 * value)
        green = int(250 - 120 * value)
        blue = int(255 - 20 * value)
        return f"rgb({red},{green},{blue})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="16" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">Bahdanau attention heatmap</text>',
    ]
    for j, token in enumerate(src_tokens):
        x = left + j * cell + cell / 2
        parts.append(
            f'<text x="{x}" y="{top - 10}" font-family="Arial, sans-serif" font-size="12" '
            f'text-anchor="end" transform="rotate(-45 {x} {top - 10})">{html.escape(token)}</text>'
        )
    for i, token in enumerate(tgt_tokens):
        y = top + i * cell + cell / 2 + 4
        parts.append(
            f'<text x="{left - 12}" y="{y}" font-family="Arial, sans-serif" font-size="12" '
            f'text-anchor="end">{html.escape(token)}</text>'
        )
        row = weights[i] if i < len(weights) else []
        for j, value in enumerate(row):
            x = left + j * cell
            y0 = top + i * cell
            parts.append(
                f'<rect x="{x}" y="{y0}" width="{cell}" height="{cell}" '
                f'fill="{color(value)}" stroke="#d1d5db" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x + cell / 2}" y="{y0 + cell / 2 + 4}" '
                f'font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#111827">{value:.2f}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate one sentence with beam search.")
    parser.add_argument("sentence", nargs="?", default="i like cats")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--save-attention", action="store_true")
    parser.add_argument("--attention-path", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device("cpu" if args.cpu else str(config["device"]))
    checkpoint_path = args.checkpoint or config["paths"]["best_checkpoint"]
    model, src_vocab, tgt_vocab, config = load_checkpoint(checkpoint_path, device)

    decoding = config["decoding"]
    result = translate_sentence(
        model,
        args.sentence,
        src_vocab,
        tgt_vocab,
        max_len=int(config["data"]["max_len"]),
        device=device,
        beam_size=args.beam_size or int(decoding["beam_size"]),
        length_penalty=float(decoding["length_penalty"]),
        max_extra_len=int(decoding["max_extra_len"]),
    )
    print(f"{args.sentence} -> {result.text}")
    print(f"beam score: {result.score:.4f}")

    if args.save_attention:
        path = args.attention_path or config["paths"]["attention_heatmap"]
        saved = save_attention_heatmap(result.attention, result.src_tokens, result.attention_tokens, path)
        print(f"attention heatmap: {saved}")


if __name__ == "__main__":
    main()
