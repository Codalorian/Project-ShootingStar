#!/usr/bin/env bash
# Phase 0 end to end. ~25 min on a 4-core laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
ATLAS_ARGS="${ATLAS_ARGS:---n-seqs 32 --seq-len 512 --batch-size 2 --cap 256}"

echo "== 1/5  roofline (measure the machine) =="
make -C bench
./bench/roofline 7 > results/roofline_raw.json 2> results/roofline.log
tail -n +2 results/roofline.log

echo; echo "== 2/5  honest baseline (the number to beat by 10x) =="
$PY bench/baseline_ollama.py --repeats 2 --n-predict 64 || \
    echo "  (skipped: ollama not running)"

echo; echo "== 3/5  redundancy atlas =="
$PY -m shootingstar.atlas.run $ATLAS_ARGS \
    --out results/atlas_tinyllama_heldout.json

echo; echo "== 4/5  quality gate self-test =="
$PY -m shootingstar.gate.selftest \
    --atlas results/atlas_tinyllama_heldout.json \
    --out results/gate_selftest.json

echo; echo "== 5/5  implied ceiling =="
$PY -m shootingstar.ceiling --shape tinyllama-1.1b \
    --atlas results/atlas_tinyllama_heldout.json

echo; echo "results/ now holds the Phase 0 record."
