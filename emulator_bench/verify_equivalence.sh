#!/usr/bin/env bash
# Checks that the MPEK wrapper matches the original code. Usage: verify_equivalence.sh [--gpu]
# Expected env: mldb (torch, torch_geometric, transformers, sentencepiece, rdkit, pytest). The script does not create envs.
# CPU mode: tiny random T5, the committed fixture and the real Mole-BERT weights; any skipped test fails the run.
# tests/test_multirun_equivalence.py (multi-run trainer against the single-run loop) runs in both modes: small model on CPU,
# full model, Triton update, CUDA graphs and a real kcat split (3 seeds, 3 epochs, about 10 min) with --gpu.
# --gpu mode: also checks CUDA graphs against the plain loop, and the real caches and splits on cuda:0. It needs the EMULaToR
# data, the built MPEK cache (cache_embeddings.py), the CataPro ProtT5 cache and the Rostlab/prot_t5_xl_uniref50 weights.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
pytest_args=(-q -rs -p no:cacheprovider --color=no -o addopts="")
out=$(CUDA_VISIBLE_DEVICES="" python -m pytest tests "${pytest_args[@]}" -m "not gpu" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
if [[ ${1:-} == --gpu ]]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m pytest tests "${pytest_args[@]}" -s -m gpu
fi
