"""KV cache wrappers for TurboQuant-style compression ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

from .quantization import TurboQuantV3


@dataclass
class CacheConfig:
    key_bits: int = 8
    value_bits: int = 4
    residual_mode: str = "fixed"  # fixed | dynamic
    residual_window: int = 128
    dynamic_fraction: float = 0.05
    dynamic_min: int = 32
    dynamic_max: int = 256
    protected_layers: int = 0
    protected_bits: int = 8
    seed: int = 42


class TurboQuantDynamicCache(DynamicCache):
    """DynamicCache that compresses old tokens and keeps a recent fp16 window."""

    def __init__(self, config: CacheConfig, n_layers: int):
        super().__init__()
        self.config = config
        self.n_layers = n_layers

        self._compressors: dict[int, TurboQuantV3] = {}
        self._chunks_k: dict[int, list[dict]] = {}
        self._chunks_v: dict[int, list[dict]] = {}
        self._total_seq: dict[int, int] = {}
        self._compressed_tokens: dict[int, int] = {}

    @staticmethod
    def _tensor_num_bytes(tensor: torch.Tensor) -> int:
        return tensor.nelement() * tensor.element_size()

    def _get_residual_window(self, total_seq: int) -> int:
        if self.config.residual_mode == "dynamic":
            rw = int(self.config.dynamic_fraction * total_seq)
            rw = max(self.config.dynamic_min, rw)
            rw = min(self.config.dynamic_max, rw)
            return max(rw, 0)
        return max(self.config.residual_window, 0)

    def _get_compressor(
        self, layer_idx: int, head_dim: int, device: torch.device
    ) -> TurboQuantV3:
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.config.key_bits,
                value_bits=self.config.value_bits,
                layer_idx=layer_idx,
                n_layers=self.n_layers,
                protected_layers=self.config.protected_layers,
                protected_bits=self.config.protected_bits,
                seed=self.config.seed,
                device=str(device),
            )
        return self._compressors[layer_idx]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        del cache_kwargs

        batch_size, n_heads, new_seq, head_dim = key_states.shape
        _ = batch_size, n_heads
        device = key_states.device

        compressor = self._get_compressor(layer_idx, head_dim, device)

        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayer())
        layer = self.layers[layer_idx]

        if layer_idx not in self._chunks_k:
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._total_seq[layer_idx] = 0
            self._compressed_tokens[layer_idx] = 0

        existing_k = getattr(layer, "keys", None)
        existing_v = getattr(layer, "values", None)
        if existing_k is None:
            full_k = key_states.contiguous()
            full_v = value_states.contiguous()
        else:
            full_k = torch.cat((existing_k, key_states), dim=2).contiguous()
            full_v = torch.cat((existing_v, value_states), dim=2).contiguous()

        layer.keys = full_k
        layer.values = full_v
        self._total_seq[layer_idx] += new_seq

        residual_window = self._get_residual_window(self._total_seq[layer_idx])
        compress_upto = max(self._total_seq[layer_idx] - residual_window, 0)
        overflow = compress_upto - self._compressed_tokens[layer_idx]

        if overflow > 0:
            start = self._compressed_tokens[layer_idx]
            end = compress_upto
            to_compress_k = full_k[:, :, start:end, :]
            to_compress_v = full_v[:, :, start:end, :]
            comp_k, comp_v = compressor.compress_kv(to_compress_k, to_compress_v)
            self._chunks_k[layer_idx].append(comp_k)
            self._chunks_v[layer_idx].append(comp_v)
            self._compressed_tokens[layer_idx] += overflow

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seq.get(layer_idx, 0)

    def get_stats(self) -> dict:
        total_tokens = sum(self._total_seq.values())
        compressed_tokens = sum(self._compressed_tokens.values())
        compressed_bytes = 0
        fp16_equivalent_bytes = 0
        materialized_bytes = 0

        for layer_idx, chunks_k in self._chunks_k.items():
            compressor = self._compressors.get(layer_idx)
            for comp_k, comp_v in zip(chunks_k, self._chunks_v[layer_idx]):
                if compressor is None:
                    continue
                mem = compressor.kv_memory_bytes(comp_k, comp_v)
                compressed_bytes += mem["compressed_bytes"]
                fp16_equivalent_bytes += mem["fp16_equivalent_bytes"]

        for layer in self.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor):
                materialized_bytes += self._tensor_num_bytes(keys)
            if isinstance(values, torch.Tensor):
                materialized_bytes += self._tensor_num_bytes(values)

        actual_total_bytes = compressed_bytes + materialized_bytes
        fp16_equivalent_total_bytes = fp16_equivalent_bytes
        return {
            "layers_seen": len(self._total_seq),
            "total_tokens": total_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_token_fraction": (
                compressed_tokens / total_tokens if total_tokens > 0 else 0.0
            ),
            "compressed_bytes": compressed_bytes,
            "fp16_equivalent_bytes": fp16_equivalent_total_bytes,
            "materialized_cache_bytes": materialized_bytes,
            "actual_total_bytes": actual_total_bytes,
            "memory_savings_ratio": (
                fp16_equivalent_total_bytes / actual_total_bytes
                if actual_total_bytes > 0
                else 0.0
            ),
            "residual_mode": self.config.residual_mode,
            "residual_window": self.config.residual_window,
            "dynamic_fraction": self.config.dynamic_fraction,
            "dynamic_min": self.config.dynamic_min,
            "dynamic_max": self.config.dynamic_max,
            "key_bits": self.config.key_bits,
            "value_bits": self.config.value_bits,
            "protected_layers": self.config.protected_layers,
            "protected_bits": self.config.protected_bits,
        }
