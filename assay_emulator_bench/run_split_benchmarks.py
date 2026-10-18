import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assay_emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    MODES,
    REPO_ROOT,
    TASK,
    discover_split_jobs,
    run_dir_for,
)

TRAIN_SCRIPT = REPO_ROOT / "assay_emulator_bench" / "train_single_split.py"


def train_command(job, seed, mode, args):
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--train_path", job["train_path"],
        "--val_path", job["val_path"],
        "--test_path", job["test_path"],
        "--embeddings_dir", args.embeddings_dir,
        "--out_dir", str(run_dir_for(Path(args.output_dir) / mode, job, seed)),
        "--mode", mode,
        "--split_group", job["split_group"],
        "--split_name", job["split_name"],
        "--device", "cuda:0" if args.gpus else "cpu",
        "--seed", str(seed),
        "--limit_rows", str(args.limit_rows),
    ]  # fmt: skip
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    return cmd + (["--overwrite"] if args.overwrite else []) + (["--no_progress"] if args.no_progress else [])


def main():
    parser = argparse.ArgumentParser(description="Run MPEK with and without pH and temperature on every kcat split, mode and seed.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR), help="Directory holding kcat/<split_group> folders.")
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR), help="Cache written by emulator_bench/cache_embeddings.py.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="*", default=["0"], help="GPU IDs. Each worker is pinned with CUDA_VISIBLE_DEVICES. Pass no ID for CPU.")
    parser.add_argument("--runs_per_gpu", type=int, default=1, help="Concurrent runs per GPU.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Smoke-test option. The default is the original 100 epochs.")
    parser.add_argument("--limit_rows", type=int, default=0, help="Smoke-test option: use only the first rows of each split file.")
    args = parser.parse_args()

    jobs = discover_split_jobs(Path(args.base_dir), tasks=[TASK], split_groups=args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    work = queue.Queue()
    for job in jobs:
        for seed in args.seeds:
            for mode in args.modes:
                work.put((job, seed, mode))
    failures, lock = [], threading.Lock()

    def worker(gpu_id):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
        while True:
            try:
                job, seed, mode = work.get_nowait()
            except queue.Empty:
                return
            if subprocess.run(train_command(job, seed, mode, args), cwd=str(REPO_ROOT), env=env).returncode:
                with lock:
                    failures.append(str(run_dir_for(Path(args.output_dir) / mode, job, seed)))

    threads = [threading.Thread(target=worker, args=(str(gpu_id),)) for gpu_id in (args.gpus or [""]) for _ in range(args.runs_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"{len(failures)} runs failed:\n" + "\n".join(failures))
    print("Run aggregate_mpek_results.py to write the summary tables.")


if __name__ == "__main__":
    main()
