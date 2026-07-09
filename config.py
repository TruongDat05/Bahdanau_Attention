from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "data": {
        "num_pairs": 1000,
        "max_len": 8,
        "train_ratio": 0.9,
    },
    "model": {
        "emb_dim": 64,
        "hidden_dim": 128,
    },
    "train": {
        "epochs": 25,
        "batch_size": 64,
        "lr": 0.001,
        "clip": 1.0,
    },
    "decoding": {
        "beam_size": 3,
        "length_penalty": 0.7,
        "max_extra_len": 2,
    },
    "paths": {
        "checkpoint_dir": "checkpoints",
        "best_checkpoint": "checkpoints/best.pt",
        "latest_checkpoint": "checkpoints/latest.pt",
        "loss_curve": "artifacts/loss_curve.svg",
        "attention_heatmap": "artifacts/attention_heatmap.svg",
    },
    "examples": [
        "i like cats",
        "we read books today",
        "she want coffee at home",
        "they need water tomorrow",
        "he love music every day",
    ],
}


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if config_path.exists():
        merge_dict(config, parse_simple_yaml(config_path.read_text(encoding="utf-8")))
    return config


def merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by config.yaml.

    This keeps the project runnable without PyYAML: nested maps use two-space
    indentation, lists use "- item", and scalar values are bool/int/float/str.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    pending_list_key: tuple[int, dict[str, Any], str] | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if line.startswith("- "):
            if pending_list_key is None:
                raise ValueError(f"List item without key: {raw_line}")
            list_indent, parent, key = pending_list_key
            if indent <= list_indent:
                raise ValueError(f"Bad list indentation: {raw_line}")
            if parent.get(key) == {}:
                parent[key] = []
            parent.setdefault(key, []).append(parse_scalar(line[2:].strip()))
            continue

        pending_list_key = None
        if ":" not in line:
            raise ValueError(f"Expected key: value line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            pending_list_key = (indent, parent, key)
        elif value == "[]":
            parent[key] = []
        else:
            parent[key] = parse_scalar(value)

    return root


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
