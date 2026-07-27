import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assay_emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_RESULTS_DIRNAME, DEFAULT_SPLIT_GROUPS, DEFAULT_TASKS, DEFAULT_SEEDS, discover_split_jobs, normalize_threshold_args, split_sizes
from assay_emulator_bench.run_split_benchmarks import CACHE_SCRIPT, maybe_cache, train_command


def load_hparams(path: str) -> dict:
    if not path:
        return {}
    with open(path) as handle:
        payload = json.load(handle)
    return payload.get("best_hparams", payload)


def apply_hparams(args, hparams: dict):
    for key in ["batch_size", "lr", "weight_decay", "clip_grad"]:
        if key in hparams:
            setattr(args, key, hparams[key])
    return args


def build_experiments(jobs, seeds, results_dirname):
    experiments = []
    for job in jobs:
        for seed in seeds:
            run_dir = Path(job["root_dir"]) / results_dirname / f"seed_{seed}"
            experiments.append({"job": job, "seed": seed, "run_dir": run_dir})
    return experiments


def run_experiment(exp, args, gpu_id):
    final_path = exp["run_dir"] / "final_results_test.csv"
    if final_path.exists() and not args.overwrite:
        status = "skipped_exists"
    else:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = train_command(exp["job"], exp["seed"], exp["run_dir"], args, "cuda:0" if args.device.startswith("cuda") else args.device)
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
        status = "completed"
    row = {"status": status, "gpu_id": str(gpu_id), "seed": exp["seed"], "run_dir": str(exp["run_dir"])}
    row.update({key: exp["job"][key] for key in ["task", "split_group", "split_name", "difficulty"]})
    if final_path.exists():
        row.update({f"test_{key}": value for key, value in pd.read_csv(final_path).iloc[0].to_dict().items()})
        assay_row_path = exp["run_dir"] / "final_results_test_assay_rows.csv"
        if assay_row_path.exists():
            row.update(
                {
                    f"test_assay_row_{key}": value
                    for key, value in pd.read_csv(assay_row_path)
                    .iloc[0]
                    .to_dict()
                    .items()
                }
            )
        row.update(split_sizes(exp["job"]))
    return row


def main():
    parser = argparse.ArgumentParser(description="Run MPEK split retraining in parallel across GPU worker slots.")
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU IDs to use. Each worker process is pinned with CUDA_VISIBLE_DEVICES=<gpu_id>.")
    parser.add_argument(
        "--runs_per_gpu",
        type=int,
        default=None,
        help="Number of concurrent retraining runs to launch per GPU.",
    )
    parser.add_argument(
        "--trials_per_gpu",
        type=int,
        default=None,
        help="Deprecated alias for --runs_per_gpu.",
    )
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--hparams_json", type=str, default=None)
    parser.add_argument("--default_settings", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--expert_dim", type=int, default=768)
    parser.add_argument("--expert_layers", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--ple_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tower_layers", type=int, default=3)
    parser.add_argument("--tower_hidden", type=int, default=128)
    parser.add_argument("--tower_dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    if args.runs_per_gpu is None:
        args.runs_per_gpu = args.trials_per_gpu if args.trials_per_gpu is not None else 1
    if args.runs_per_gpu <= 0:
        raise ValueError("--runs_per_gpu must be positive")
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    apply_hparams(args, load_hparams(args.hparams_json))
    maybe_cache(args)
    jobs = discover_split_jobs(Path(args.base_dir), tasks=args.tasks, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    work = queue.Queue()
    for exp in build_experiments(jobs, args.seeds, args.results_dirname):
        work.put(exp)

    rows, lock = [], threading.Lock()

    def worker(gpu_id, slot_index):
        while True:
            try:
                exp = work.get_nowait()
            except queue.Empty:
                return
            try:
                row = run_experiment(exp, args, gpu_id)
                row["slot_index"] = slot_index
            except Exception as exc:
                row = {"status": "failed", "gpu_id": str(gpu_id), "slot_index": slot_index, "error": str(exc), "run_dir": str(exp["run_dir"])}
                row.update({key: exp["job"][key] for key in ["task", "split_group", "split_name", "difficulty"]})
            with lock:
                rows.append(row)
            work.task_done()

    threads = []
    for gpu_id in args.gpus:
        for slot_index in range(args.runs_per_gpu):
            thread = threading.Thread(target=worker, args=(gpu_id, slot_index), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    out_path = Path(args.base_dir) / "mpek_parallel_retrain_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"{len(failed)} retrain jobs failed")


if __name__ == "__main__":
    main()
