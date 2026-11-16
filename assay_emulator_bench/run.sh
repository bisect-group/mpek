#!/usr/bin/env bash
# Reproduce the MPEK pH/T assay study: caches, train and test both modes on P-seq and Time splits, summary.
# Usage: assay_emulator_bench/run.sh [--dry-run] [prepare|cache|train|aggregate|all]
# Env: PY, GPUS (one id), RUNS_PER_GPU, SEEDS, DATA_ROOT, OUT_ROOT, EMB_ROOT, DRY_RUN=1.
# train also evaluates; finished runs are skipped. The cache is shared with emulator_bench.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=${PY:-python}
GPUS=${GPUS:-0}
RUNS_PER_GPU=${RUNS_PER_GPU:-4}
read -ra SEEDS <<<"${SEEDS:-666 777 888}"
DATA=() OUT=() EMB=()
[[ -z ${DATA_ROOT:-} ]] || DATA=(--base_dir "$DATA_ROOT")
[[ -z ${OUT_ROOT:-} ]] || OUT=(--output_dir "$OUT_ROOT")
[[ -z ${EMB_ROOT:-} ]] || EMB=(--embeddings_dir "$EMB_ROOT")
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    prepare | cache | train | aggregate | all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [prepare|cache|train|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

if want prepare; then
  run "$PY" emulator_bench/prepare_assets.py
fi
if want cache; then
  run env CUDA_VISIBLE_DEVICES="$GPUS" "$PY" emulator_bench/cache_embeddings.py --device cuda:0 --tasks kcat
fi
if want train; then
  run "$PY" assay_emulator_bench/run_split_benchmarks.py --gpus "$GPUS" --runs_per_gpu "$RUNS_PER_GPU" \
    --seeds "${SEEDS[@]}" "${DATA[@]}" "${EMB[@]}" "${OUT[@]}"
fi
if want aggregate; then
  run "$PY" assay_emulator_bench/aggregate_mpek_results.py "${OUT[@]}"
fi
