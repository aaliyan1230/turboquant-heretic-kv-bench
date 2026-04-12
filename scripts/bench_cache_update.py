#!/usr/bin/env python3
"""Microbenchmark TurboQuant cache update overhead across chunk sizes."""

from __future__ import annotations

import argparse
import time

import torch

from tqhk.cache import CacheConfig, TurboQuantDynamicCache


def run_case(
    chunk_size: int,
    *,
    prefill_tokens: int,
    decode_steps: int,
    n_heads: int,
    head_dim: int,
    residual_window: int,
    dtype: torch.dtype,
) -> dict:
    cache = TurboQuantDynamicCache(
        CacheConfig(
            key_bits=8,
            value_bits=4,
            residual_mode="fixed",
            residual_window=residual_window,
            compression_chunk_size=chunk_size,
        ),
        n_layers=1,
    )

    prefill_k = torch.randn(1, n_heads, prefill_tokens, head_dim, dtype=dtype)
    prefill_v = torch.randn(1, n_heads, prefill_tokens, head_dim, dtype=dtype)
    cache.update(prefill_k, prefill_v, 0)

    start = time.perf_counter()
    for _ in range(decode_steps):
        step_k = torch.randn(1, n_heads, 1, head_dim, dtype=dtype)
        step_v = torch.randn(1, n_heads, 1, head_dim, dtype=dtype)
        cache.update(step_k, step_v, 0)
    elapsed = time.perf_counter() - start

    stats = cache.get_stats()
    return {
        "chunk_size": chunk_size,
        "decode_update_sec": elapsed,
        "compressed_tokens": int(stats["compressed_tokens"]),
        "pending_uncompressed_tokens": int(stats["pending_uncompressed_tokens"]),
        "compression_token_fraction": float(stats["compression_token_fraction"]),
        "estimated_compression_ratio": float(stats["estimated_compression_ratio"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-tokens", type=int, default=1024)
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--residual-window", type=int, default=128)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    args = parser.parse_args()

    dtype = torch.float16
    print(
        "prefill_tokens=", args.prefill_tokens,
        " decode_steps=", args.decode_steps,
        " residual_window=", args.residual_window,
        sep="",
    )
    print(
        "chunk_size,decode_update_sec,compressed_tokens,pending_uncompressed_tokens,compression_fraction,estimated_ratio"
    )

    for chunk_size in args.chunk_sizes:
        row = run_case(
            chunk_size,
            prefill_tokens=args.prefill_tokens,
            decode_steps=args.decode_steps,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            residual_window=args.residual_window,
            dtype=dtype,
        )
        print(
            f"{row['chunk_size']},{row['decode_update_sec']:.6f},{row['compressed_tokens']},"
            f"{row['pending_uncompressed_tokens']},{row['compression_token_fraction']:.4f},"
            f"{row['estimated_compression_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()