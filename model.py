from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class Encoder(nn.Module):
    """Bi-GRU encoder: x_1...x_T -> h_1...h_T."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, pad_id: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_bridge = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(src_ids)
        encoder_outputs, hidden = self.gru(embedded)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        decoder_hidden = torch.tanh(self.hidden_bridge(hidden_cat)).unsqueeze(0)
        return encoder_outputs, decoder_hidden


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention."""

    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        self.score_projection = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_encoder = self.encoder_projection(encoder_outputs)
        projected_decoder = self.decoder_projection(decoder_hidden).unsqueeze(1)
        attention_scores = self.score_projection(
            torch.tanh(projected_encoder + projected_decoder)
        ).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~src_mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return attention_scores, attention_weights, context


@dataclass
class DecoderOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    attention_scores: torch.Tensor
    attention_weights: torch.Tensor
    context_vectors: torch.Tensor
    decoder_outputs: torch.Tensor


class Decoder(nn.Module):
    """GRU decoder with a fresh Bahdanau context vector at every step."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attention = AdditiveAttention(encoder_hidden_dim, decoder_hidden_dim, decoder_hidden_dim)
        self.gru = nn.GRU(emb_dim + encoder_hidden_dim, decoder_hidden_dim, batch_first=True)
        self.output = nn.Linear(decoder_hidden_dim + encoder_hidden_dim, vocab_size)

    def forward(
        self,
        tgt_in_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_hidden: torch.Tensor,
        src_mask: torch.Tensor,
        decoder_hidden: torch.Tensor | None = None,
    ) -> DecoderOutput:
        embedded = self.embedding(tgt_in_ids)
        if decoder_hidden is None:
            decoder_hidden = encoder_hidden

        logits_steps: list[torch.Tensor] = []
        score_steps: list[torch.Tensor] = []
        weight_steps: list[torch.Tensor] = []
        context_steps: list[torch.Tensor] = []
        decoder_steps: list[torch.Tensor] = []

        for t in range(tgt_in_ids.size(1)):
            previous_state = decoder_hidden[-1]
            attention_scores, attention_weights, context = self.attention(
                previous_state,
                encoder_outputs,
                src_mask,
            )
            rnn_input = torch.cat([embedded[:, t], context], dim=-1).unsqueeze(1)
            decoder_output, decoder_hidden = self.gru(rnn_input, decoder_hidden)
            decoder_output = decoder_output.squeeze(1)
            logits = self.output(torch.cat([decoder_output, context], dim=-1))

            logits_steps.append(logits)
            score_steps.append(attention_scores)
            weight_steps.append(attention_weights)
            context_steps.append(context)
            decoder_steps.append(decoder_output)

        return DecoderOutput(
            logits=torch.stack(logits_steps, dim=1),
            hidden=decoder_hidden,
            attention_scores=torch.stack(score_steps, dim=1),
            attention_weights=torch.stack(weight_steps, dim=1),
            context_vectors=torch.stack(context_steps, dim=1),
            decoder_outputs=torch.stack(decoder_steps, dim=1),
        )


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pad_id: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_id = src_pad_id

    def forward(self, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor) -> torch.Tensor:
        src_mask = src_ids.ne(self.src_pad_id)
        encoder_outputs, encoder_hidden = self.encoder(src_ids)
        decoder_output = self.decoder(tgt_in_ids, encoder_outputs, encoder_hidden, src_mask)
        return decoder_output.logits


def build_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_pad_id: int,
    tgt_pad_id: int,
    emb_dim: int,
    hidden_dim: int,
) -> Seq2Seq:
    encoder = Encoder(src_vocab_size, emb_dim, hidden_dim, src_pad_id)
    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        emb_dim=emb_dim,
        encoder_hidden_dim=hidden_dim * 2,
        decoder_hidden_dim=hidden_dim,
        pad_id=tgt_pad_id,
    )
    return Seq2Seq(encoder, decoder, src_pad_id)
