"""Compatibility shims for transformers >= 5.x.

The vendored IndexTTS generation code was written for transformers 4.x and
imports several internal utilities that were removed, renamed, or moved in
v5.x.  This module provides drop-in replacements so the code continues to
work without modifying its business logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Weight-name constants (removed from transformers.utils in 5.x)
# ---------------------------------------------------------------------------

FLAX_WEIGHTS_NAME = "flax_model.msgpack"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "tf_model.h5"


# ---------------------------------------------------------------------------
# is_offline_mode (moved / renamed in 5.x)
# ---------------------------------------------------------------------------

try:
    from transformers.utils.hub import is_offline_mode
except ImportError:
    try:
        from huggingface_hub import is_offline_mode
    except ImportError:
        def is_offline_mode() -> bool:
            return os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"


# ---------------------------------------------------------------------------
# is_remote_url / download_url (removed in 5.x)
# ---------------------------------------------------------------------------

def is_remote_url(url_or_filename: str) -> bool:
    """Return True if *url_or_filename* is a remote URL (transformers 4.x compat)."""
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def download_url(url: str, *args, **kwargs) -> str:
    """Stub — IndexTTS never downloads weights at inference time."""
    raise NotImplementedError(
        "download_url is not implemented in the 5.x compat shim. "
        "IndexTTS should never call this at inference time."
    )


# ---------------------------------------------------------------------------
# is_safetensors_available / is_torch_sdpa_available (removed in 5.x)
# ---------------------------------------------------------------------------

def is_safetensors_available() -> bool:
    """Return True if the safetensors library is importable."""
    try:
        import safetensors  # noqa: F401
        return True
    except ImportError:
        return False


def is_torch_sdpa_available() -> bool:
    """Return True if PyTorch SDPA (scaled dot-product attention) is available."""
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")


# ---------------------------------------------------------------------------
# isin_mps_friendly (removed from transformers.pytorch_utils in 5.x)
# ---------------------------------------------------------------------------

# torch.isin already handles MPS correctly in modern PyTorch.
isin_mps_friendly = torch.isin


# ---------------------------------------------------------------------------
# pytorch_utils — head-pruning helpers (removed in 5.x)
# ---------------------------------------------------------------------------

def find_pruneable_heads_and_indices(
    heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
):
    """Find heads to prune and their index ranges (transformers 4.x compat)."""
    heads_to_prune = set(heads) - already_pruned_heads
    if not heads_to_prune:
        return heads, {}
    heads_indices = {}
    for head in sorted(heads_to_prune):
        n_pruned = sum(1 for h in already_pruned_heads if h < head)
        idx = (head - n_pruned) * head_size
        heads_indices[head] = idx
    return list(heads_to_prune), heads_indices


def prune_conv1d_layer(layer: nn.Module, index: torch.Tensor, dim: int = 0):
    """Prune a Conv1D layer (transformers 4.x compat)."""
    if dim == 0:
        layer.weight = nn.Parameter(layer.weight.index_select(0, index))
        if layer.bias is not None:
            layer.bias = nn.Parameter(layer.bias.index_select(0, index))
    elif dim == 1:
        layer.weight = nn.Parameter(layer.weight.index_select(1, index))


def prune_layer(
    layer: nn.Module,
    heads_to_prune: list[int],
    head_size: int,
    heads_indices: dict[int, int],
    already_pruned_heads: set[int],
    num_heads: int,
    dim: int = 0,
):
    """Prune attention heads from a layer (transformers 4.x compat).

    IndexTTS inference never calls pruning — this is a no-op stub for compat.
    """
    already_pruned_heads.update(heads_to_prune)


# ---------------------------------------------------------------------------
# model_parallel_utils — device-map helpers (removed in 5.x)
# ---------------------------------------------------------------------------

def get_device_map(n_layers: int, devices) -> dict[int, int]:
    """Return a simple device map spreading layers across devices.

    The original transformers 4.x helper was more sophisticated (handled
    pipelining, device IDs, etc.) but IndexTTS only ever uses the result
    to count covered layers.  A single-device map is sufficient.
    """
    return {i: 0 for i in range(n_layers)}


def assert_device_map(device_map: dict, n_layers: int) -> None:
    """Validate that *device_map* covers all *n_layers* (transformers 4.x compat)."""
    covered = set(device_map.keys())
    expected = set(range(n_layers))
    if not expected.issubset(covered):
        missing = expected - covered
        raise ValueError(
            f"Device map does not cover all layers — missing: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# SequenceSummary — GPT-2 pooling head (removed from modeling_utils in 5.x)
# ---------------------------------------------------------------------------

class SequenceSummary(nn.Module):
    """GPT-2 sequence summary / pooling layer (transformers 4.x compat).

    A single linear layer that produces a fixed-size vector from the
    last hidden state, matching the original GPT-2 multiple-choice head.
    """

    def __init__(self, config):
        super().__init__()
        self.summary = nn.Linear(config.n_embd, 1)

    def forward(self, hidden_states):
        output = self.summary(hidden_states)   # (batch, seq_len, 1)
        output = output.squeeze(-1)            # (batch, seq_len)
        output = output.mean(dim=-1)           # (batch,)
        return output


# ---------------------------------------------------------------------------
# QuantizedCacheConfig — removed from transformers.cache_utils in 5.x
# ---------------------------------------------------------------------------

@dataclass
class QuantizedCacheConfig:
    backend: str = "quanto"
    nbits: int = 4
    axis_key: int = 0
    axis_value: int = 0
    q_group_size: int = 64
    residual_length: int = 128


# ---------------------------------------------------------------------------
# ExtensionsTrie — removed from transformers.tokenization_utils in 5.x
# ---------------------------------------------------------------------------

class ExtensionsTrie:
    """Minimal trie for vocabulary prefix search (token healing)."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab
        self._trie: dict = {}
        for token_str in vocab:
            node = self._trie
            for ch in token_str:
                node = node.setdefault(ch, {})
            node[""] = token_str

    def extensions(self, prefix: str) -> list[str]:
        node = self._trie
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        results: list[str] = []
        self._collect(node, results)
        return results

    def _collect(self, node: dict, results: list[str]) -> None:
        for key, child in node.items():
            if key == "":
                results.append(child)
            else:
                self._collect(child, results)
