# Bahdanau Attention for Toy Machine Translation

This repository implements a compact sequence-to-sequence neural machine
translation model with Bahdanau additive attention. The task is a synthetic
English-to-Vietnamese translation problem designed for learning and inspecting
the mechanics of encoder-decoder models rather than for large-scale benchmark
performance.

The dataset contains short parallel sentence pairs such as:

```text
i like cats              -> toi thich meo
we read books today      -> chung toi doc sach hom nay
she want coffee at home  -> co ay muon ca phe o nha
```

## Approach

The model uses a bidirectional GRU encoder to read the full source sentence and
produce contextual source states. A GRU decoder then generates the translation
token by token. At each decoding step, Bahdanau attention scores every source
state against the current decoder state, normalizes those scores into attention
weights, and builds a context vector for the next target token.

```text
e_tj     = v_a^T tanh(W_s s_{t-1} + W_h h_j)
alpha_tj = softmax(e_tj)
c_t      = sum_j alpha_tj h_j
```

This makes the alignment behavior visible: the saved attention heatmap shows
which source tokens the decoder focused on while producing each target token.

## Features

- Lightweight synthetic translation dataset
- Bidirectional GRU encoder
- GRU decoder with Bahdanau additive attention
- Beam search inference
- Checkpoint saving for latest, best, and per-epoch models
- Validation loss, token accuracy, and corpus BLEU
- SVG attention heatmap
- SVG training loss curve
- YAML configuration tuned for small GPUs

## Project Layout

```text
dataset.py      Synthetic corpus, Dataset, and DataLoader helpers
vocab.py        Vocabulary and special token utilities
model.py        Encoder, attention layer, decoder, and Seq2Seq model
train.py        Training loop, checkpointing, BLEU, and artifacts
translate.py    Beam search inference and attention visualization
evaluate.py     Validation metrics and BLEU evaluation
config.yaml     Default experiment configuration
```

## Quick Start

Install the only required dependency:

```bash
pip install -r requirements.txt
```

Train with the default lightweight configuration:

```bash
python train.py
```

Run a short CPU smoke test:

```bash
python train.py --epochs 3 --cpu
```

Evaluate a checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

Translate a sentence with beam search:

```bash
python translate.py "we read books today" --checkpoint checkpoints/best.pt
```

Save an attention heatmap for a translation:

```bash
python translate.py "i like cats" --checkpoint checkpoints/best.pt --save-attention
```

## Outputs

Training writes model checkpoints to `checkpoints/` and visual artifacts to
`artifacts/`:

```text
checkpoints/best.pt
checkpoints/latest.pt
checkpoints/epoch_*.pt
checkpoints/history.json
artifacts/loss_curve.svg
artifacts/attention_heatmap.svg
```

The default configuration keeps memory usage small enough for a 4GB GPU:

```yaml
data:
  num_pairs: 1000
  max_len: 8
model:
  emb_dim: 64
  hidden_dim: 128
train:
  batch_size: 64
decoding:
  beam_size: 3
```

## Notes

The implementation intentionally avoids heavy external dependencies. Loss
curves and attention heatmaps are written directly as SVG files, and BLEU is
computed locally for this toy corpus. `toy_seq2seq_no_attention.py` is kept only
as a compatibility wrapper for older commands.
