"""
Evaluation text.

Redundancy is a property of a model *on a distribution*. Measuring it on
wikitext and then deploying on code or chat is exactly the calibration-set
overfitting that makes published pruning numbers fail to reproduce in
practice. So: `held_out` is the neutral reference, and `workload` is whatever
you actually care about -- drop .txt files in data/ and they get used.
"""
from __future__ import annotations

import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Last-resort corpus so the harness always runs. Deliberately mixed-genre.
_FALLBACK = [
    "The transformer architecture processes sequences by attending over all "
    "previous positions. During autoregressive generation, each new token "
    "requires a full forward pass through every layer of the network.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    p = xs[len(xs)//2]\n"
    "    lo = [x for x in xs if x < p]\n    hi = [x for x in xs if x > p]\n"
    "    return quicksort(lo) + [x for x in xs if x == p] + quicksort(hi)",
    "In 1687 Isaac Newton published the Principia Mathematica, which set out "
    "the laws of motion and universal gravitation. The work reframed celestial "
    "mechanics as a consequence of a single inverse-square law.",
    "User: my laptop fan spins up whenever I run the model. Assistant: that is "
    "expected -- sustained matrix multiplication saturates the CPU and the "
    "package hits its thermal limit within about ninety seconds.",
    "The mitochondrion generates most of the cell's supply of adenosine "
    "triphosphate through oxidative phosphorylation across the inner membrane.",
]


def _clean(paragraphs: list[str], min_chars: int) -> list[str]:
    out = []
    for p in paragraphs:
        p = p.strip()
        if len(p) < min_chars or p.startswith("="):
            continue
        out.append(re.sub(r"\s+", " ", p))
    return out


def held_out(n_docs: int = 256, min_chars: int = 400) -> list[str]:
    """Neutral reference text: wikitext-2 test split, cached via HF hub."""
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        fp = hf_hub_download("Salesforce/wikitext",
                             "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                             repo_type="dataset")
        texts = pq.read_table(fp).column("text").to_pylist()
        docs = _clean(texts, min_chars)
        if docs:
            return docs[:n_docs]
    except Exception as e:  # offline, or dataset moved
        print(f"[corpus] held_out fallback ({type(e).__name__}: {e})")
    return (_FALLBACK * ((n_docs // len(_FALLBACK)) + 1))[:n_docs]


def workload(n_docs: int = 256, min_chars: int = 200) -> list[str]:
    """Your actual traffic. Drop .txt files into data/ to define it."""
    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        return []
    chunks: list[str] = []
    for f in files:
        chunks += _clean(f.read_text(errors="ignore").split("\n\n"), min_chars)
    return chunks[:n_docs]


def get(split: str = "held_out", **kw) -> list[str]:
    docs = {"held_out": held_out, "workload": workload}[split](**kw)
    if not docs:
        raise SystemExit(f"corpus '{split}' is empty -- add .txt files to {DATA_DIR}")
    return docs
