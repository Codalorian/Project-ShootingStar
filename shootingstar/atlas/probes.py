"""
The redundancy atlas.

Five probes, each answering "how many bytes did this component NOT need to
move for this token?" -- because bytes/token, not FLOPs, is what bounds
decoding (see shootingstar/hw.py).

  DepthProbe           layers that are near-identity maps
  MLPWidthProbe        neurons that contribute nothing for THIS token
  AttnHeadProbe        heads that contribute nothing for THIS token
  ResidualRankProbe    is the residual stream actually d-dimensional?
                       (centred -- the uncentred spectrum just finds the DC offset)
  TokenDifficultyProbe how many tokens are easy enough to draft (spec-decoding)

Every truncation number is an EXACT reconstruction error, not a proxy: we
rebuild the sublayer output from the top-k components and measure relative
L2 error against the true output. Proxies (activation magnitude, weight norm)
systematically overstate how prunable a model is, because they ignore
cancellation between the terms you keep.
"""
from __future__ import annotations

import math
import torch

KEEP_FRACS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90)


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


class _Reservoir:
    """Keep at most `cap` token-vectors, strided across batches."""

    def __init__(self, cap: int):
        self.cap, self.buf, self.aux, self.n = cap, [], [], 0

    def __init_aux(self):
        pass

    def add(self, x: torch.Tensor, aux: torch.Tensor | None = None):   # x: (N, D)
        if self.n >= self.cap:
            return
        take = min(self.cap - self.n, x.shape[0])
        step = max(1, x.shape[0] // take)
        idx = torch.arange(0, x.shape[0], step)[:take]
        self.buf.append(x[idx].detach().to(torch.float32).clone())
        if aux is not None and aux.shape[0] == x.shape[0]:
            self.aux.append(aux[idx].detach().to(torch.float32).clone())
        self.n += idx.numel()

    def tensor(self):
        return torch.cat(self.buf, 0) if self.buf else None

    def aux_tensor(self):
        if not self.aux or sum(a.shape[0] for a in self.aux) != self.n:
            return None
        return torch.cat(self.aux, 0)


def _truncation_curve(contrib_score: torch.Tensor, rebuild, full: torch.Tensor,
                      resid_norm: torch.Tensor | None = None):
    """Relative L2 error of the output rebuilt from the top-k scoring parts.

    contrib_score: (N, C) importance of each of C parts, per token
    rebuild(mask) -> (N, D) output using only the masked parts
    full:          (N, D) true output
    """
    N, C = contrib_score.shape
    order = contrib_score.argsort(dim=1, descending=True)
    fnorm = full.norm(dim=1).clamp_min(1e-9)
    out = {}
    for frac in KEEP_FRACS:
        k = max(1, int(round(frac * C)))
        mask = torch.zeros(N, C, dtype=torch.bool)
        mask.scatter_(1, order[:, :k], True)
        absdiff = (rebuild(mask) - full).norm(dim=1)
        err = absdiff / fnorm
        entry = {
            "rel_err_mean": err.mean().item(),
            "rel_err_p95": err.quantile(0.95).item(),
        }
        if resid_norm is not None:
            rerr = absdiff / resid_norm.clamp_min(1e-9)
            entry["rel_err_vs_resid_mean"] = rerr.mean().item()
            entry["rel_err_vs_resid_p95"] = rerr.quantile(0.95).item()
        out[frac] = entry
    return out


class ResidualContext:
    """Per-layer input residual norms, so sublayer errors can be expressed as a
    fraction of the signal that actually propagates down the network."""

    def __init__(self, model):
        self.norms: dict[int, torch.Tensor] = {}
        self.handles = [l.register_forward_pre_hook(self._mk(i), with_kwargs=True)
                        for i, l in enumerate(model.model.layers)]

    def _mk(self, i):
        def hook(mod, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states")
            if torch.is_tensor(x):
                self.norms[i] = x.detach().float().reshape(-1, x.shape[-1]).norm(dim=1)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------
class DepthProbe:
    """Angular distance between a layer's input and output residual stream.

    A layer whose output points in nearly the same direction as its input is
    doing nearly nothing; Gromov et al. show these cluster in the deeper-middle
    and are the first candidates for removal.
    """

    def __init__(self, model):
        self.layers = model.model.layers
        self.stats = [{"cos": [], "dnorm": [], "attn_ratio": [], "mlp_ratio": []}
                      for _ in self.layers]
        self.handles = []
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(
                self._mk_layer_hook(i), with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(
                self._mk_sub_hook(i, "attn_ratio")))
            self.handles.append(layer.mlp.register_forward_hook(
                self._mk_sub_hook(i, "mlp_ratio")))
        self._resid_in = {}

    @staticmethod
    def _first_tensor(x):
        if torch.is_tensor(x):
            return x
        if isinstance(x, (tuple, list)):
            for e in x:
                if torch.is_tensor(e):
                    return e
        return None

    def _mk_layer_hook(self, i):
        def hook(mod, args, kwargs, output):
            x_in = self._first_tensor(args) if args else kwargs.get("hidden_states")
            x_out = self._first_tensor(output)
            if x_in is None or x_out is None:
                return
            a = x_in.detach().float().reshape(-1, x_in.shape[-1])
            b = x_out.detach().float().reshape(-1, x_out.shape[-1])
            self._resid_in[i] = a.norm(dim=1)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
            self.stats[i]["cos"].append(cos.mean().item())
            self.stats[i]["dnorm"].append(
                ((b - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-9)).mean().item())
        return hook

    def _mk_sub_hook(self, i, key):
        def hook(mod, args, output):
            y = self._first_tensor(output)
            r = self._resid_in.get(i)
            if y is None or r is None:
                return
            y = y.detach().float().reshape(-1, y.shape[-1])
            if y.shape[0] != r.shape[0]:
                return
            self.stats[i][key].append(
                (y.norm(dim=1) / r.clamp_min(1e-9)).mean().item())
        return hook

    def result(self):
        out = []
        for i, s in enumerate(self.stats):
            c = _mean(s["cos"])
            out.append({
                "layer": i,
                "cos_in_out": c,
                "angular_distance": math.acos(max(-1.0, min(1.0, c))) / math.pi,
                "delta_norm_ratio": _mean(s["dnorm"]),
                "attn_out_over_resid": _mean(s["attn_ratio"]),
                "mlp_out_over_resid": _mean(s["mlp_ratio"]),
            })
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------
class MLPWidthProbe:
    """Per-token contextual sparsity of the MLP.

    Captures h = act(gate(x)) * up(x), the input to down_proj. Each neuron i
    contributes h_i * W_down[:, i] to the output. If only a small fraction of
    neurons carry the output for any given token, then the rows of gate/up and
    columns of down for the rest never needed to be read from memory.
    """

    def __init__(self, model, cap: int = 192, ctx: "ResidualContext | None" = None):
        self.layers = model.model.layers
        self.ctx = ctx
        self.res = [_Reservoir(cap) for _ in self.layers]
        self.sparsity = [[] for _ in self.layers]
        self.handles = [
            l.mlp.down_proj.register_forward_pre_hook(self._mk(i))
            for i, l in enumerate(self.layers)]

    def _mk(self, i):
        def hook(mod, args):
            h = args[0].detach().float().reshape(-1, args[0].shape[-1])
            self.res[i].add(h, self.ctx.norms.get(i) if self.ctx else None)
            a = h.abs()
            thr = a.max(dim=1, keepdim=True).values * 0.01
            self.sparsity[i].append((a < thr).float().mean().item())
        return hook

    def result(self):
        out = []
        for i, layer in enumerate(self.layers):
            h = self.res[i].tensor()
            if h is None:
                continue
            W = layer.mlp.down_proj.weight.detach().float()   # (hidden, inter)
            colnorm = W.norm(dim=0)                            # (inter,)
            full = h @ W.T
            score = h.abs() * colnorm
            curve = _truncation_curve(score, lambda m: (h * m) @ W.T, full,
                                      self.res[i].aux_tensor())
            out.append({
                "layer": i,
                "n_neurons": h.shape[1],
                "frac_below_1pct_of_max": _mean(self.sparsity[i]),
                "truncation": curve,
            })
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------
class AttnHeadProbe:
    """Per-token head redundancy, measured at o_proj's input."""

    def __init__(self, model, cap: int = 192, ctx: "ResidualContext | None" = None):
        self.layers = model.model.layers
        self.ctx = ctx
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.res = [_Reservoir(cap) for _ in self.layers]
        self.handles = [
            l.self_attn.o_proj.register_forward_pre_hook(self._mk(i))
            for i, l in enumerate(self.layers)]

    def _mk(self, i):
        def hook(mod, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            self.res[i].add(x, self.ctx.norms.get(i) if self.ctx else None)
        return hook

    def result(self):
        out = []
        for i, layer in enumerate(self.layers):
            x = self.res[i].tensor()
            if x is None:
                continue
            W = layer.self_attn.o_proj.weight.detach().float()   # (hidden, heads*hd)
            H, hd = self.n_heads, self.head_dim
            if x.shape[1] != H * hd:
                continue
            full = x @ W.T
            xh = x.view(-1, H, hd)
            # exact per-head contribution norm
            Wh = W.view(W.shape[0], H, hd)
            contrib = torch.einsum("nhd,ohd->nho", xh, Wh)       # (N, H, hidden)
            score = contrib.norm(dim=2)                          # (N, H)

            def rebuild(mask):                                   # mask (N, H)
                return (contrib * mask.unsqueeze(-1)).sum(dim=1)

            curve = _truncation_curve(score, rebuild, full, self.res[i].aux_tensor())
            energy = score.mean(dim=0)
            out.append({
                "layer": i,
                "n_heads": H,
                "head_energy_norm": (energy / energy.sum()).tolist(),
                "truncation": curve,
            })
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------
class ResidualRankProbe:
    """Effective dimensionality of the residual stream, via a streaming Gram."""

    def __init__(self, model):
        self.layers = model.model.layers
        d = model.config.hidden_size
        self.gram = [torch.zeros(d, d, dtype=torch.float64) for _ in self.layers]
        self.sum = [torch.zeros(d, dtype=torch.float64) for _ in self.layers]
        self.count = [0] * len(self.layers)
        self.handles = [l.register_forward_hook(self._mk(i))
                        for i, l in enumerate(self.layers)]

    def _mk(self, i):
        def hook(mod, args, output):
            y = DepthProbe._first_tensor(output)
            if y is None:
                return
            y = y.detach().double().reshape(-1, y.shape[-1])
            self.gram[i] += y.T @ y
            self.sum[i] += y.sum(dim=0)
            self.count[i] += y.shape[0]
        return hook

    @staticmethod
    def _spectrum(M):
        ev = torch.linalg.eigvalsh(M).flip(0).clamp_min(0)
        p = ev / ev.sum().clamp_min(1e-30)
        cum = p.cumsum(0)
        return {
            "participation_ratio": (1.0 / (p ** 2).sum()).item(),
            "dims_for_90pct": int((cum < 0.90).sum().item()) + 1,
            "dims_for_99pct": int((cum < 0.99).sum().item()) + 1,
            "dims_for_999pct": int((cum < 0.999).sum().item()) + 1,
            "top1_share": p[0].item(),
        }

    def result(self):
        out = []
        for i, G in enumerate(self.gram):
            n = self.count[i]
            if not n:
                continue
            second = G / n
            mu = self.sum[i] / n
            cov = second - torch.outer(mu, mu)
            # symmetrise against float drift before eigendecomposition
            cov = 0.5 * (cov + cov.T)
            out.append({
                "layer": i,
                "dim": G.shape[0],
                "n_tokens": n,
                # a covariance from n samples has rank <= n-1; below ~4x the
                # hidden size the spectrum is undersampling, not structure.
                "undersampled": bool(n < 4 * G.shape[0]),
                "mean_norm_over_rms": (mu.norm() / second.diagonal().sum().sqrt()
                                       .clamp_min(1e-30)).item(),
                "uncentred": self._spectrum(second),
                "centred": self._spectrum(cov),
            })
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------
class TokenDifficultyProbe:
    """How much headroom does speculative decoding have?

    Speculative decoding wins when a cheap drafter can guess the big model's
    token. Its ceiling is set by how peaked the big model's own distribution
    is: a token whose top-1 probability is ~1.0 is trivially draftable.
    Unlike every other probe here, exploiting this is LOSSLESS.
    """

    def __init__(self, thresholds=(0.5, 0.8, 0.9, 0.95, 0.99)):
        self.thresholds = thresholds
        self.top1, self.ent, self.top5 = [], [], []
        self.easy_runs = {t: [] for t in thresholds}

    def observe(self, logits: torch.Tensor):
        """logits: (T, V) for one sequence, in order."""
        lp = torch.log_softmax(logits.float(), dim=-1)
        p = lp.exp()
        top = p.topk(5, dim=-1).values
        self.top1 += top[:, 0].tolist()
        self.top5 += top.sum(dim=1).tolist()
        self.ent += (-(p * lp).sum(dim=-1)).tolist()
        for t in self.thresholds:
            easy = (top[:, 0] > t).tolist()
            run = 0
            for e in easy:
                if e:
                    run += 1
                else:
                    self.easy_runs[t].append(run)
                    run = 0
            self.easy_runs[t].append(run)

    def result(self):
        import statistics as st
        out = {
            "n_tokens": len(self.top1),
            "top1_prob_mean": _mean(self.top1),
            "top1_prob_median": st.median(self.top1) if self.top1 else float("nan"),
            "top5_mass_mean": _mean(self.top5),
            "entropy_nats_mean": _mean(self.ent),
            "easy_fraction": {}, "mean_easy_run": {},
        }
        for t in self.thresholds:
            out["easy_fraction"][t] = _mean([x > t for x in self.top1])
            runs = [r for r in self.easy_runs[t]]
            out["mean_easy_run"][t] = _mean(runs)
        return out


# --------------------------------------------------------------------------
class Atlas:
    """Runs all hook-based probes together in a single forward pass."""

    def __init__(self, model, cap: int = 192, rank: bool = True):
        self.ctx = ResidualContext(model)
        self.depth = DepthProbe(model)
        self.mlp = MLPWidthProbe(model, cap, self.ctx)
        self.attn = AttnHeadProbe(model, cap, self.ctx)
        self.rank = ResidualRankProbe(model) if rank else None
        self.tok = TokenDifficultyProbe()

    def result(self):
        r = {
            "depth": self.depth.result(),
            "mlp_width": self.mlp.result(),
            "attn_heads": self.attn.result(),
            "token_difficulty": self.tok.result(),
            "keep_fracs": list(KEEP_FRACS),
        }
        if self.rank:
            r["residual_rank"] = self.rank.result()
        return r

    def remove(self):
        for p in (self.ctx, self.depth, self.mlp, self.attn, self.rank):
            if p is not None:
                p.remove()
