"""
Roofline analysis for single-stream LLM decoding.

The governing fact: generating one token at batch size 1 reads every weight
exactly once and does ~2 ops per weight. That fixes the arithmetic intensity
at ~2/bytes_per_weight ops/byte, independent of model size. Compare it to the
machine balance (peak ops / peak bandwidth) to see how much of the machine is
idle, and therefore which kind of redundancy is worth exploiting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


# --------------------------------------------------------------------------
# Machine
# --------------------------------------------------------------------------
@dataclass
class Machine:
    read_bw_gbps: float          # sustained DRAM read, at the best thread count
    best_bw_threads: int
    fp32_gflops: float
    int8_gops: float
    compute_threads: int
    cache_bw_gbps: dict = field(default_factory=dict)   # bytes -> GB/s
    source: str = "measured"

    @classmethod
    def from_roofline_json(cls, path: str | Path) -> "Machine":
        d = json.loads(Path(path).read_text())
        bw = {int(k): v["median"] for k, v in d["read_bw_gbps"].items()}
        best_t = max(bw, key=bw.get)
        fl = {int(k): v["median"] for k, v in d["fp32_fma_gflops"].items()}
        i8 = {int(k): v["median"] for k, v in d["int8_vnni_gops"].items()}
        ct = max(i8, key=i8.get)
        sweep = {int(k): v for k, v in d.get("read_bw_size_sweep_gbps", {}).items()}
        return cls(
            read_bw_gbps=bw[best_t], best_bw_threads=best_t,
            fp32_gflops=fl[max(fl, key=fl.get)], int8_gops=i8[ct],
            compute_threads=ct, cache_bw_gbps=sweep,
        )

    @property
    def balance_int8(self) -> float:
        """ops/byte needed to saturate the int8 units."""
        return self.int8_gops / self.read_bw_gbps

    @property
    def balance_fp32(self) -> float:
        return self.fp32_gflops / self.read_bw_gbps

    def bw_at_working_set(self, nbytes: int) -> float:
        """Interpolate achievable read bandwidth for a working set of nbytes.

        This is the lever the cache hierarchy hands us: a working set that fits
        in L2/L3 streams an order of magnitude faster than one that does not.
        """
        if not self.cache_bw_gbps:
            return self.read_bw_gbps
        pts = sorted(self.cache_bw_gbps.items())
        if nbytes <= pts[0][0]:
            return pts[0][1]
        for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
            if nbytes <= s1:
                # log-linear interpolation across the tier boundary
                import math
                f = (math.log(nbytes) - math.log(s0)) / (math.log(s1) - math.log(s0))
                return b0 + f * (b1 - b0)
        return pts[-1][1]


# --------------------------------------------------------------------------
# Model byte accounting
# --------------------------------------------------------------------------
@dataclass
class ModelShape:
    name: str
    n_layers: int
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    vocab: int
    head_dim: int = 0

    def __post_init__(self):
        if not self.head_dim:
            self.head_dim = self.hidden // self.n_heads

    @classmethod
    def from_hf_config(cls, cfg, name: str = "") -> "ModelShape":
        get = lambda *ks: next(getattr(cfg, k) for k in ks if getattr(cfg, k, None))
        return cls(
            name=name or getattr(cfg, "_name_or_path", "model"),
            n_layers=get("num_hidden_layers"),
            hidden=get("hidden_size"),
            intermediate=get("intermediate_size"),
            n_heads=get("num_attention_heads"),
            n_kv_heads=getattr(cfg, "num_key_value_heads", None) or get("num_attention_heads"),
            vocab=get("vocab_size"),
            head_dim=getattr(cfg, "head_dim", 0) or 0,
        )

    # ---- per-component parameter counts touched by ONE decode step ----
    def params_per_layer(self) -> dict[str, int]:
        h, i = self.hidden, self.intermediate
        kv = self.n_kv_heads * self.head_dim
        q = self.n_heads * self.head_dim
        return {
            "attn.q_proj": h * q,
            "attn.k_proj": h * kv,
            "attn.v_proj": h * kv,
            "attn.o_proj": q * h,
            "mlp.gate_proj": h * i,
            "mlp.up_proj": h * i,
            "mlp.down_proj": i * h,
        }

    def params_breakdown(self) -> dict[str, int]:
        per = self.params_per_layer()
        out = {k: v * self.n_layers for k, v in per.items()}
        out["lm_head"] = self.hidden * self.vocab
        return out

    def decode_params(self) -> int:
        return sum(self.params_breakdown().values())

    def weight_bytes_per_token(self, bits: float = 4.0) -> int:
        return int(self.decode_params() * bits / 8)

    def kv_bytes_per_token(self, ctx_len: int, bits: float = 16.0) -> int:
        """KV cache bytes RE-READ for every generated token at context ctx_len."""
        per_tok_per_layer = 2 * self.n_kv_heads * self.head_dim
        return int(per_tok_per_layer * self.n_layers * ctx_len * bits / 8)

    def flops_per_token(self) -> int:
        return 2 * self.decode_params()


# --------------------------------------------------------------------------
# The bound
# --------------------------------------------------------------------------
@dataclass
class DecodeBound:
    tok_s_bandwidth: float
    tok_s_compute: float
    tok_s_ceiling: float
    bound_by: str
    arithmetic_intensity: float
    machine_balance: float
    idle_compute_frac: float
    bytes_per_token: int
    weight_bytes: int
    kv_bytes: int


def decode_bound(m: Machine, s: ModelShape, bits: float = 4.0,
                 ctx_len: int = 0, kv_bits: float = 16.0,
                 working_set_bytes: int | None = None) -> DecodeBound:
    wb = s.weight_bytes_per_token(bits)
    kvb = s.kv_bytes_per_token(ctx_len, kv_bits) if ctx_len else 0
    total_b = wb + kvb
    flops = s.flops_per_token()

    bw = m.bw_at_working_set(working_set_bytes) if working_set_bytes else m.read_bw_gbps
    tok_bw = bw * 1e9 / total_b
    # int8 path is the relevant compute peak for quantised inference
    tok_cp = m.int8_gops * 1e9 / flops
    ai = flops / total_b
    return DecodeBound(
        tok_s_bandwidth=tok_bw, tok_s_compute=tok_cp,
        tok_s_ceiling=min(tok_bw, tok_cp),
        bound_by="bandwidth" if tok_bw < tok_cp else "compute",
        arithmetic_intensity=ai, machine_balance=m.balance_int8,
        idle_compute_frac=max(0.0, 1.0 - ai / m.balance_int8),
        bytes_per_token=total_b, weight_bytes=wb, kv_bytes=kvb,
    )


KNOWN_SHAPES = {
    "tinyllama-1.1b": ModelShape("tinyllama-1.1b", 22, 2048, 5632, 32, 4, 32000),
    "llama-3.2-1b":   ModelShape("llama-3.2-1b", 16, 2048, 8192, 32, 8, 128256, head_dim=64),
    "llama-3.2-3b":   ModelShape("llama-3.2-3b", 28, 3072, 8192, 24, 8, 128256, head_dim=128),
    "llama-2-7b":     ModelShape("llama-2-7b", 32, 4096, 11008, 32, 32, 32000),
    "qwen3-4b":       ModelShape("qwen3-4b", 36, 2560, 9728, 32, 8, 151936, head_dim=128),
}
