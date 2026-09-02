"""
The quality gate.

"Retain 95% intelligence" has to be an operational number BEFORE we have a
stake in the answer, or every result becomes a negotiation.

Primary metric: mean KL( P_reference || P_modified ) per token, in nats.
Why not the usual choices:

  * Multiple-choice accuracy (MMLU et al.) is a rank statistic over 4 options.
    A model can lose most of its calibration and generative coherence while
    its argmax over 4 lettered options survives. It is the least sensitive
    instrument commonly used and it is why "free 40% pruning" results so often
    fail to reproduce in generation.
  * Perplexity on ground-truth text only scores the probability of ONE token.
    It is blind to how the other 31,999 got rearranged, and rearranging those
    is exactly what pruning does.

KL scores the entire distribution against the model we are trying to preserve,
at every position. It is continuous, cheap, and hard to game. We report the
insensitive metrics alongside it precisely so the gap is visible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
class _Bypass(nn.Module):
    """Identity stand-in for a decoder layer.

    We bypass rather than delete so that layer_idx-based KV cache indexing
    stays valid. This measures the QUALITY cost of removing a layer; the
    speed benefit is accounted for separately in hw.py.
    """

    def __init__(self, orig: nn.Module):
        super().__init__()
        self.orig = orig
        self._tuple_out = False

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,) if self._tuple_out else hidden_states


def bypass_layers(model, idxs) -> dict:
    """Replace the given decoder layers with identities. Returns undo info."""
    layers = model.model.layers
    saved = {}
    for i in idxs:
        saved[i] = layers[i]
        b = _Bypass(layers[i])
        # match whatever container the real layer returns
        b._tuple_out = getattr(model, "_ss_tuple_out", False)
        layers[i] = b
    return saved


def restore_layers(model, saved: dict) -> None:
    for i, layer in saved.items():
        model.model.layers[i] = layer


def detect_layer_output_shape(model, sample_ids) -> bool:
    """True if decoder layers return a tuple (transformers version dependent)."""
    box = {}

    def hook(mod, args, output):
        box["tuple"] = isinstance(output, tuple)

    h = model.model.layers[0].register_forward_hook(hook)
    try:
        model(sample_ids[:1, :8])
    finally:
        h.remove()
    model._ss_tuple_out = box.get("tuple", False)
    return model._ss_tuple_out


# --------------------------------------------------------------------------
@dataclass
class GateResult:
    kl_mean: float            # PRIMARY: nats, KL(ref || mod)
    kl_median: float
    kl_p95: float
    top1_agreement: float     # fraction of positions with same argmax
    top5_overlap: float       # mean |top5_ref ∩ top5_mod| / 5
    ppl_ref: float            # perplexity on ground truth -- insensitive
    ppl_mod: float
    ppl_ratio: float
    acc_ref: float            # ground-truth next-token argmax accuracy
    acc_mod: float            # same, modified -- a RANK statistic, like MC accuracy
    n_tokens: int

    def passes(self, kl_budget: float = 0.05, agree_budget: float = 0.95) -> bool:
        return self.kl_mean <= kl_budget and self.top1_agreement >= agree_budget

    def dict(self):
        return asdict(self)


class ReferenceLogits:
    """Cached reference log-probabilities, so many variants can be scored
    against one unmodified forward pass."""

    def __init__(self, model, batches: torch.Tensor, batch_size: int = 2,
                 store_dtype=torch.float16):
        self.logprobs, self.targets = [], []
        with torch.no_grad():
            for i in range(0, batches.shape[0], batch_size):
                ch = batches[i: i + batch_size]
                lg = model(ch).logits[:, :-1, :]
                self.logprobs.append(torch.log_softmax(lg.float(), -1).to(store_dtype))
                self.targets.append(ch[:, 1:])
        self.logprobs = torch.cat(self.logprobs, 0)
        self.targets = torch.cat(self.targets, 0)

    @property
    def n_tokens(self) -> int:
        return self.targets.numel()


def _ppl(logprobs: torch.Tensor, targets: torch.Tensor) -> float:
    ll = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    return float(torch.exp(-ll.mean()))


def evaluate(model, ref: ReferenceLogits, batches: torch.Tensor,
             batch_size: int = 2) -> GateResult:
    kls, agree, overlap = [], [], []
    mod_lp_for_ppl, acc_ref, acc_mod = [], [], []
    with torch.no_grad():
        for i in range(0, batches.shape[0], batch_size):
            ch = batches[i: i + batch_size]
            lg = model(ch).logits[:, :-1, :]
            q = torch.log_softmax(lg.float(), -1)                    # modified
            p = ref.logprobs[i: i + batch_size].float()              # reference
            pe = p.exp()
            kl = (pe * (p - q)).sum(-1)                              # KL(ref||mod)
            kls.append(kl.reshape(-1))
            agree.append((p.argmax(-1) == q.argmax(-1)).reshape(-1).float())
            tp = p.topk(5, -1).indices
            tq = q.topk(5, -1).indices
            hit = (tp.unsqueeze(-1) == tq.unsqueeze(-2)).any(-1).float().sum(-1) / 5.0
            overlap.append(hit.reshape(-1))
            mod_lp_for_ppl.append(
                q.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1).reshape(-1))
            gt = ch[:, 1:]
            acc_ref.append((p.argmax(-1) == gt).reshape(-1).float())
            acc_mod.append((q.argmax(-1) == gt).reshape(-1).float())

    kl = torch.cat(kls)
    ppl_ref = _ppl(ref.logprobs.float(), ref.targets)
    ppl_mod = float(torch.exp(-torch.cat(mod_lp_for_ppl).mean()))
    return GateResult(
        kl_mean=kl.mean().item(),
        kl_median=kl.median().item(),
        kl_p95=kl.quantile(0.95).item(),
        top1_agreement=torch.cat(agree).mean().item(),
        top5_overlap=torch.cat(overlap).mean().item(),
        ppl_ref=ppl_ref, ppl_mod=ppl_mod, ppl_ratio=ppl_mod / ppl_ref,
        acc_ref=torch.cat(acc_ref).mean().item(),
        acc_mod=torch.cat(acc_mod).mean().item(),
        n_tokens=int(kl.numel()),
    )


def sample_generation(model, tok, prompt: str, max_new: int = 48) -> str:
    """Greedy sample -- the eyeball check that numbers can't replace."""
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
