"""TurboQuant-style quantization primitives used by the benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
from scipy import integrate


def generate_rotation_matrix(
    dim: int,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    gaussian = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(gaussian)

    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(0)
    return q.to(device)


def _gaussian_approx_pdf(x: float, dim: int) -> float:
    sigma2 = 1.0 / dim
    return (1.0 / math.sqrt(2.0 * math.pi * sigma2)) * math.exp(
        -(x * x) / (2.0 * sigma2)
    )


def _solve_lloyd_max(
    dim: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bits > 4:
        sigma = 1.0 / math.sqrt(dim)
        centroids = torch.linspace(
            -4.0 * sigma, 4.0 * sigma, 2**bits, dtype=torch.float32
        )
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        return centroids, boundaries

    n_levels = 2**bits
    sigma = 1.0 / math.sqrt(dim)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3.0] + boundaries + [hi * 3.0]

        new_centroids = []
        for i in range(n_levels):
            left, right = edges[i], edges[i + 1]
            num, _ = integrate.quad(
                lambda x: x * _gaussian_approx_pdf(x, dim),
                left,
                right,
            )
            den, _ = integrate.quad(
                lambda x: _gaussian_approx_pdf(x, dim),
                left,
                right,
            )
            if den > 1e-15:
                new_centroids.append(num / den)
            else:
                new_centroids.append(centroids[i])

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


@lru_cache(maxsize=128)
def _get_codebook(dim: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _solve_lloyd_max(dim=dim, bits=bits)


@dataclass
class MSECompressor:
    """Single-stage MSE compressor with bit-packed storage."""

    head_dim: int
    bits: int
    seed: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.rotation = generate_rotation_matrix(
            self.head_dim,
            seed=self.seed,
            device=self.device,
        )
        self.centroids = _get_codebook(self.head_dim, self.bits)[0].to(self.device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        bsz, n_heads, seq_len, head_dim = states.shape
        n_vec = bsz * n_heads * seq_len
        flat = states.reshape(n_vec, head_dim).float()

        norms = torch.norm(flat, dim=-1)
        flat_norm = flat / (norms.unsqueeze(-1) + 1e-8)

        rotated = flat_norm @ self.rotation.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        indices_per_byte = 8 // self.bits
        pad = (indices_per_byte - (head_dim % indices_per_byte)) % indices_per_byte

        packed_input = indices.long()
        if pad:
            packed_input = F.pad(packed_input, (0, pad))

        n_groups = packed_input.shape[-1] // indices_per_byte
        powers = torch.tensor(
            [2 ** (self.bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=packed_input.device,
        )
        packed = (
            (packed_input.reshape(n_vec, n_groups, indices_per_byte) * powers)
            .sum(dim=-1)
            .to(torch.uint8)
        )

        return {
            "idx_bytes": packed.reshape(bsz, n_heads, seq_len, n_groups),
            "vec_norms": norms.to(torch.float16).reshape(bsz, n_heads, seq_len),
            "shape": (bsz, n_heads, seq_len, head_dim),
            "idx_pad": pad,
            "dtype_bytes": states.element_size(),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        bsz, n_heads, seq_len, head_dim = compressed["shape"]
        n_vec = bsz * n_heads * seq_len

        idx_bytes = compressed["idx_bytes"].reshape(n_vec, -1)
        vec_norms = compressed["vec_norms"].reshape(n_vec, 1).float()
        pad = compressed["idx_pad"]

        indices_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        shifts = torch.tensor(
            [self.bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_bytes.device,
        )
        indices = ((idx_bytes.long().unsqueeze(-1) >> shifts) & mask).reshape(n_vec, -1)
        if pad:
            indices = indices[:, :head_dim]

        reconstructed = (self.centroids[indices] @ self.rotation) * vec_norms
        return reconstructed.reshape(bsz, n_heads, seq_len, head_dim)

    def compressed_num_bytes(self, compressed: dict) -> int:
        total = 0
        for key in ("idx_bytes", "vec_norms"):
            tensor = compressed.get(key)
            if isinstance(tensor, torch.Tensor):
                total += tensor.nelement() * tensor.element_size()
        return total

    def fp_equivalent_num_bytes(self, compressed: dict) -> int:
        bsz, n_heads, seq_len, head_dim = compressed["shape"]
        dtype_bytes = int(compressed.get("dtype_bytes", 2))
        return bsz * n_heads * seq_len * head_dim * dtype_bytes


class TurboQuantV3:
    """TurboQuant-style asymmetric K/V compressor with optional layer protection."""

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 8,
        value_bits: int = 4,
        layer_idx: int = 0,
        n_layers: int = 32,
        protected_layers: int = 0,
        protected_bits: int = 8,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        protected = layer_idx < protected_layers or layer_idx >= (
            n_layers - protected_layers
        )
        effective_key_bits = min(protected_bits if protected else key_bits, 8)
        effective_value_bits = min(protected_bits if protected else value_bits, 8)

        self.key_bits = effective_key_bits
        self.value_bits = effective_value_bits

        base_seed = seed + (layer_idx * 1000)
        self.key_compressor = MSECompressor(
            head_dim=head_dim,
            bits=self.key_bits,
            seed=base_seed,
            device=device,
        )
        self.value_compressor = MSECompressor(
            head_dim=head_dim,
            bits=self.value_bits,
            seed=base_seed + 500,
            device=device,
        )

    @torch.no_grad()
    def compress_kv(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[dict, dict]:
        return self.key_compressor.compress(keys), self.value_compressor.compress(
            values
        )

    @torch.no_grad()
    def decompress_kv(
        self, comp_keys: dict, comp_values: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.key_compressor.decompress(comp_keys),
            self.value_compressor.decompress(comp_values),
        )

    def kv_memory_bytes(self, comp_keys: dict, comp_values: dict) -> dict:
        compressed_bytes = self.key_compressor.compressed_num_bytes(
            comp_keys
        ) + self.value_compressor.compressed_num_bytes(comp_values)
        fp16_equivalent_bytes = self.key_compressor.fp_equivalent_num_bytes(
            comp_keys
        ) + self.value_compressor.fp_equivalent_num_bytes(comp_values)
        return {
            "compressed_bytes": compressed_bytes,
            "fp16_equivalent_bytes": fp16_equivalent_bytes,
            "compression_ratio": (
                fp16_equivalent_bytes / compressed_bytes
                if compressed_bytes > 0
                else 0.0
            ),
        }
