import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assay_emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_SEEDS,
    DEFAULT_TASKS,
)
from assay_emulator_bench.run_split_benchmarks import maybe_cache
from assay_emulator_bench.tune_optuna import metric_direction, prepare_storage

TUNE_SCRIPT = REPO_ROOT / "assay_emulator_bench" / "tune_optuna.py"


def split_trials(total, workers):
    base = total // workers
    remainder = total % workers
    return [base + (1 if idx < remainder else 0) for idx in range(workers)]


def worker_cmd(args, trials, worker_index):
    cmd = [
        sys.executable,
        str(TUNE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--tasks",
        *args.tasks,
        "--split_groups",
        *args.split_groups,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--ph_col",
        args.ph_col,
        "--temperature_col",
        args.temperature_col,
        "--temperature_unit",
        args.temperature_unit,
        "--pair_reducer",
        args.pair_reducer,
        "--device",
        "cuda:0" if args.device.startswith("cuda") else args.device,
        "--cache_device",
        "cuda:0" if args.cache_device.startswith("cuda") else args.cache_device,
        "--epochs",
        str(args.epochs),
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--metric",
        args.metric,
        "--eval_split",
        args.eval_split,
        "--n_trials",
        str(trials),
        "--study_name",
        args.study_name,
        "--storage",
        args.storage,
        "--sampler_seed",
        str(args.sampler_seed + worker_index),
        "--skip_cache",
    ]
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.seeds:
        cmd.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    for flag in ["persistent_workers", "pin_memory", "preload", "overwrite_runs"]:
        if getattr(args, flag):
            cmd.append("--" + flag)
    if args.use_assay_conditions:
        cmd.append("--use_assay_conditions")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch parallel Optuna workers for the MPEK bench.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(DEFAULT_EMBEDDINGS_DIR),
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--use_assay_conditions", action="store_true")
    parser.add_argument("--ph_col", type=str, default="pH")
    parser.add_argument("--temperature_col", type=str, default="Temperature")
    parser.add_argument(
        "--temperature_unit",
        choices=["kelvin", "celsius"],
        default="kelvin",
    )
    parser.add_argument(
        "--pair_reducer",
        choices=["none", "max"],
        default="none",
    )
    parser.add_argument("--prottrans_path", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--protein_batch_size", type=int, default=1)
    parser.add_argument("--ligand_batch_size", type=int, default=256)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--study_name", type=str, default="mpek_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--reset_storage", action="store_true")
    parser.add_argument("--stagger_seconds", type=float, default=3.0)
    args = parser.parse_args()

    if args.storage is None:
        args.storage = "sqlite:///%s" % (Path(args.base_dir) / "optuna_studies" / f"{args.study_name}.db")
    maybe_cache(args)
    prepare_storage(args)
    optuna.create_study(direction=metric_direction(args.metric), study_name=args.study_name, storage=args.storage, load_if_exists=True)
    slots = [(gpu, slot) for gpu in args.gpus for slot in range(args.trials_per_gpu)]
    trial_counts = split_trials(args.n_trials, len(slots))
    processes = []
    try:
        for worker_index, ((gpu_id, slot), trials) in enumerate(zip(slots, trial_counts)):
            if trials <= 0:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Launching Optuna worker {worker_index} on GPU {gpu_id} slot {slot} for {trials} trials", flush=True)
            proc = subprocess.Popen(worker_cmd(args, trials, worker_index), cwd=str(REPO_ROOT), env=env)
            processes.append(proc)
            if args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)
        failed = False
        for proc in processes:
            failed = proc.wait() != 0 or failed
        if failed:
            raise RuntimeError("One or more Optuna workers failed.")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
