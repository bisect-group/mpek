import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import REPO_ROOT, run_dir_for
from emulator_bench.run_split_benchmarks import add_common_args, discover_jobs, maybe_cache, train_command


def main():
    parser = argparse.ArgumentParser(description="Run MPEK with its original settings on every discovered split and seed across GPU worker slots.")
    add_common_args(parser)
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU IDs. Each worker is pinned with CUDA_VISIBLE_DEVICES=<gpu_id>.")
    parser.add_argument("--runs_per_gpu", type=int, default=1, help="Concurrent runs per GPU.")
    args = parser.parse_args()
    args.cache_device = args.cache_device or f"cuda:{args.gpus[0]}"

    jobs = discover_jobs(args)
    maybe_cache(args)
    work = queue.Queue()
    for job in jobs:
        for seed in args.seeds:
            work.put((job, seed))
    failures, lock = [], threading.Lock()

    def worker(gpu_id):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        while True:
            try:
                job, seed = work.get_nowait()
            except queue.Empty:
                return
            result = subprocess.run(train_command(job, seed, args, "cuda:0"), cwd=str(REPO_ROOT), env=env)
            if result.returncode:
                with lock:
                    failures.append(str(run_dir_for(args.output_dir, job, seed)))

    threads = [threading.Thread(target=worker, args=(gpu_id,)) for gpu_id in args.gpus for _ in range(args.runs_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"{len(failures)} runs failed:\n" + "\n".join(failures))
    print("Run aggregate_mpek_results.py to write the summary tables.")


if __name__ == "__main__":
    main()
