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
        self._recent_k: dict[int, list[torch.Tensor]] = {}
        self._recent_v: dict[int, list[torch.Tensor]] = {}
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

        if layer_idx not in self._chunks_k:
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._recent_k[layer_idx] = []
            self._recent_v[layer_idx] = []
            self._total_seq[layer_idx] = 0
            self._compressed_tokens[layer_idx] = 0

        self._total_seq[layer_idx] += new_seq

        self._recent_k[layer_idx].append(key_states)
        self._recent_v[layer_idx].append(value_states)

        recent_k = torch.cat(self._recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._recent_v[layer_idx], dim=2)

        residual_window = self._get_residual_window(self._total_seq[layer_idx])
        if residual_window == 0:
            overflow = recent_k.shape[2]
        else:
            overflow = max(recent_k.shape[2] - residual_window, 0)

        if overflow > 0:
            to_compress_k = recent_k[:, :, :overflow, :]
            to_compress_v = recent_v[:, :, :overflow, :]
            comp_k, comp_v = compressor.compress_kv(to_compress_k, to_compress_v)
            self._chunks_k[layer_idx].append(comp_k)
            self._chunks_v[layer_idx].append(comp_v)
            self._compressed_tokens[layer_idx] += overflow

            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._recent_k[layer_idx] = [recent_k]
            self._recent_v[layer_idx] = [recent_v]

        parts_k = []
        parts_v = []

        for comp_k, comp_v in zip(self._chunks_k[layer_idx], self._chunks_v[layer_idx]):
            deq_k, deq_v = compressor.decompress_kv(comp_k, comp_v)
            parts_k.append(deq_k.to(key_states.dtype))
            parts_v.append(deq_v.to(value_states.dtype))

        recent_k = torch.cat(self._recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._recent_v[layer_idx], dim=2)
        parts_k.append(recent_k)
        parts_v.append(recent_v)

        # SDPA expects the last dimension to be contiguous.
        if len(parts_k) == 1:
            full_k = parts_k[0].contiguous()
            full_v = parts_v[0].contiguous()
        else:
            full_k = torch.cat(parts_k, dim=2).contiguous()
            full_v = torch.cat(parts_v, dim=2).contiguous()

        while len(self.layers) <= layer_idx:
            self.layers.append(DynamicLayer())

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seq.get(layer_idx, 0)

    def get_stats(self) -> dict:
        total_tokens = sum(self._total_seq.values())
        compressed_tokens = sum(self._compressed_tokens.values())
        compressed_bytes = 0
        fp16_equivalent_bytes = 0
        recent_bytes = 0

        for layer_idx, chunks_k in self._chunks_k.items():
            compressor = self._compressors.get(layer_idx)
            for comp_k, comp_v in zip(chunks_k, self._chunks_v[layer_idx]):
                if compressor is None:
                    continue
                mem = compressor.kv_memory_bytes(comp_k, comp_v)
                compressed_bytes += mem["compressed_bytes"]
                fp16_equivalent_bytes += mem["fp16_equivalent_bytes"]

        for recent_parts in self._recent_k.values():
            recent_bytes += sum(self._tensor_num_bytes(tensor) for tensor in recent_parts)
        for recent_parts in self._recent_v.values():
            recent_bytes += sum(self._tensor_num_bytes(tensor) for tensor in recent_parts)

        actual_total_bytes = compressed_bytes + recent_bytes
        fp16_equivalent_total_bytes = fp16_equivalent_bytes + recent_bytes
        return {
            "layers_seen": len(self._total_seq),
            "total_tokens": total_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_token_fraction": (
                compressed_tokens / total_tokens if total_tokens > 0 else 0.0
            ),
            "compressed_bytes": compressed_bytes,
            "fp16_equivalent_bytes": fp16_equivalent_total_bytes,
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
