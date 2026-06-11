"""Compatibility shims for transformers >= 5.x.

The vendored IndexTTS generation code was written for transformers 4.x and
imports several internal utilities that were removed, renamed, or moved in
v5.x.  Each section below first attempts to import from the original
transformers location (so 4.x still works unchanged), then falls back to a
local re-implementation for 5.x.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Weight-name constants (removed from transformers.utils in 5.x)
# ---------------------------------------------------------------------------

try:
    from transformers.utils import FLAX_WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME
except ImportError:
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

try:
    from transformers.utils import is_remote_url, download_url
except ImportError:
    def is_remote_url(url_or_filename: str) -> bool:
        """Return True if *url_or_filename* is a remote URL."""
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

try:
    from transformers.utils import is_safetensors_available, is_torch_sdpa_available
except ImportError:
    def is_safetensors_available() -> bool:
        try:
            import safetensors  # noqa: F401
            return True
        except ImportError:
            return False

    def is_torch_sdpa_available() -> bool:
        return hasattr(torch.nn.functional, "scaled_dot_product_attention")


# ---------------------------------------------------------------------------
# isin_mps_friendly (removed from transformers.pytorch_utils in 5.x)
# ---------------------------------------------------------------------------

try:
    from transformers.pytorch_utils import isin_mps_friendly
except ImportError:
    # torch.isin handles MPS correctly in modern PyTorch.
    isin_mps_friendly = torch.isin


# ---------------------------------------------------------------------------
# pytorch_utils — head-pruning helpers (removed in 5.x)
# ---------------------------------------------------------------------------

try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_conv1d_layer
except ImportError:
    def find_pruneable_heads_and_indices(
        heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
    ) -> tuple[set[int], torch.LongTensor]:
        """Return (heads_to_prune, index_tensor_to_keep) — matches transformers 4.x signature."""
        heads = set(heads) - already_pruned_heads
        mask = torch.ones(n_heads, head_size)
        # Adjust head indices to account for already-pruned heads
        adjusted = set(
            head - sum(1 if h < head else 0 for h in already_pruned_heads)
            for head in heads
        )
        mask[list(adjusted)] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index: torch.LongTensor = torch.arange(len(mask))[mask].long()
        return heads, index

    def prune_conv1d_layer(layer: nn.Module, index: torch.Tensor, dim: int = 1) -> nn.Module:
        """Prune a Conv1D layer in-place and return it (transformers 4.x compat)."""
        if dim == 0:
            layer.weight = nn.Parameter(layer.weight.index_select(0, index))
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias.index_select(0, index))
        elif dim == 1:
            layer.weight = nn.Parameter(layer.weight.index_select(1, index))
        return layer


try:
    from transformers.modeling_utils import prune_layer
except ImportError:
    def prune_layer(layer, index, dim=None):
        """No-op stub — IndexTTS inference never calls head pruning."""
        return layer


# ---------------------------------------------------------------------------
# model_parallel_utils — device-map helpers (removed in 5.x)
# ---------------------------------------------------------------------------

try:
    from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
except ImportError:
    def get_device_map(n_layers: int, devices) -> dict:
        """Distribute *n_layers* evenly across *devices* (transformers 4.x compat)."""
        devices = list(devices)
        if not devices:
            return {0: list(range(n_layers))}
        layers_per_device = math.ceil(n_layers / len(devices))
        device_map: dict = {}
        layer_idx = 0
        for device in devices:
            bucket = list(range(layer_idx, min(layer_idx + layers_per_device, n_layers)))
            if bucket:
                device_map[device] = bucket
            layer_idx += layers_per_device
        return device_map

    def assert_device_map(device_map: dict, n_layers: int) -> None:
        """Validate that *device_map* covers all *n_layers* (transformers 4.x compat)."""
        covered: set[int] = set()
        for layers in device_map.values():
            if isinstance(layers, (list, range)):
                covered.update(layers)
            else:
                covered.add(layers)
        expected = set(range(n_layers))
        if not expected.issubset(covered):
            missing = expected - covered
            raise ValueError(
                f"Device map does not cover all layers — missing: {sorted(missing)}"
            )


# ---------------------------------------------------------------------------
# SequenceSummary — GPT-2 pooling head (removed from modeling_utils in 5.x)
# ---------------------------------------------------------------------------

try:
    from transformers.modeling_utils import SequenceSummary
except ImportError:
    class SequenceSummary(nn.Module):
        """GPT-2 sequence summary / pooling layer (transformers 4.x compat).

        Mirrors the original transformers implementation: a linear projection
        over the last hidden state with optional activation and first/last/mean
        pooling.  IndexTTS only uses the 'last' (default) summary type.
        """

        def __init__(self, config):
            super().__init__()
            self.summary_type = getattr(config, "summary_type", "last")
            num_labels = getattr(config, "num_labels", 1)
            self.summary = nn.Linear(config.n_embd, num_labels)
            self.first_dropout = nn.Dropout(
                getattr(config, "summary_first_dropout", 0.0)
            )
            self.last_dropout = nn.Dropout(
                getattr(config, "summary_last_dropout", 0.0)
            )

        def forward(self, hidden_states, cls_index=None):
            if self.summary_type == "last":
                output = hidden_states[:, -1]
            elif self.summary_type == "first":
                output = hidden_states[:, 0]
            elif self.summary_type == "mean":
                output = hidden_states.mean(dim=1)
            elif self.summary_type == "cls_index":
                if cls_index is None:
                    cls_index = torch.full_like(
                        hidden_states[..., :1, :],
                        hidden_states.shape[-2] - 1,
                        dtype=torch.long,
                    )
                else:
                    cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                    cls_index = cls_index.expand(
                        (-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),)
                    )
                output = hidden_states.gather(-2, cls_index).squeeze(-2)
            else:
                raise ValueError(f"Unsupported summary_type: {self.summary_type}")

            output = self.first_dropout(output)
            output = self.summary(output)
            output = self.last_dropout(output)
            return output


# ---------------------------------------------------------------------------
# QuantizedCacheConfig — removed from transformers.cache_utils in 5.x
# ---------------------------------------------------------------------------

try:
    from transformers.cache_utils import QuantizedCacheConfig
except ImportError:
    @dataclass
    class QuantizedCacheConfig:
        backend: str = "quanto"
        nbits: int = 4
        axis_key: int = 0
        axis_value: int = 0
        q_group_size: int = 64
        residual_length: int = 128


# ---------------------------------------------------------------------------
# OffloadedCache — removed from transformers.cache_utils in 5.x
# ---------------------------------------------------------------------------

try:
    from transformers.cache_utils import OffloadedCache
except ImportError:
    from transformers.cache_utils import DynamicCache as OffloadedCache


# ---------------------------------------------------------------------------
# ExtensionsTrie — removed from transformers.tokenization_utils in 5.x
# ---------------------------------------------------------------------------

try:
    from transformers.tokenization_utils import ExtensionsTrie
except ImportError:
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


# ---------------------------------------------------------------------------
# Beam-search cache reordering (transformers 5.x uses Cache objects)
# ---------------------------------------------------------------------------

def reorder_past_key_values(past, beam_idx):
    """Reorder KV cache for beam search (legacy tuple + transformers 5 Cache)."""
    try:
        from transformers.cache_utils import Cache
    except ImportError:
        Cache = None

    if Cache is not None and isinstance(past, Cache):
        past.reorder_cache(beam_idx)
        return past

    return tuple(
        tuple(
            past_state.index_select(0, beam_idx.to(past_state.device))
            if past_state is not None
            else None
            for past_state in layer_past
        )
        for layer_past in past
    )
