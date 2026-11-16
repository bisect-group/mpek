#!/usr/bin/env bash
# Reproduce MPEK: asset check, caches, train and test kcat and km (all splits, seeds), seed summary.
# Usage: emulator_bench/run.sh [--dry-run] [prepare|cache|train|aggregate|all]
# Env: PY, GPUS (one id), SEEDS, DATA_ROOT, OUT_ROOT, DRY_RUN=1.
# train also evaluates; finished runs are skipped and unfinished runs resume.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
read -ra SEEDS <<<"${SEEDS:-666 777 888}"
DATA=() OUT=()
[[ -z ${DATA_ROOT:-} ]] || DATA=(--base_dir "$DATA_ROOT")
[[ -z ${OUT_ROOT:-} ]] || OUT=(--output_dir "$OUT_ROOT")
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
  run "$PY" emulator_bench/cache_embeddings.py --device cuda:0 "${DATA[@]}" "${OUT[@]}"
fi
if want train; then
  run "$PY" emulator_bench/launch_multirun.py --skip_cache --tasks kcat km --group_size 1 --seeds "${SEEDS[@]}" "${DATA[@]}" "${OUT[@]}"
fi
if want aggregate; then
  run "$PY" emulator_bench/aggregate_mpek_results.py "${OUT[@]}"
fi
