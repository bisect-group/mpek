import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_TASKS,
    REPO_ROOT,
    discover_split_jobs,
    run_dir_for,
)

BENCH_DIR = REPO_ROOT / "emulator_bench"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR), help="Directory holding <task>/<split_group> folders.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Receives embeddings/ and <task>/ run folders.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--cache_device", type=str, default=None, help="Device for the embedding cache step. Default: the first run device.")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Opt in to mixed precision (bf16 on Ampere or newer, else fp16). The original trains in fp32.")
    parser.add_argument("--epochs", type=int, default=None, help="Smoke-test option. The default is the original 100 epochs.")
    parser.add_argument("--limit_rows", type=int, default=0, help="Smoke-test option: use only the first rows of each split file.")


def discover_jobs(args):
    jobs = discover_split_jobs(Path(args.base_dir), tasks=args.tasks, split_groups=args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    return jobs


def maybe_cache(args) -> None:
    if args.skip_cache:
        return
    cmd = [
        sys.executable, str(BENCH_DIR / "cache_embeddings.py"),
        "--base_dir", args.base_dir,
        "--output_dir", args.output_dir,
        "--tasks", *args.tasks,
        "--split_groups", *args.split_groups,
        "--device", args.cache_device,
        "--limit_rows", str(args.limit_rows),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def train_command(job, seed, args, device):
    cmd = [
        sys.executable, str(BENCH_DIR / "train_single_split.py"),
        "--train_path", job["train_path"],
        "--val_path", job["val_path"],
        "--test_path", job["test_path"],
        "--embeddings_dir", str(Path(args.output_dir) / "embeddings"),
        "--out_dir", str(run_dir_for(args.output_dir, job, seed)),
        "--task", job["task"],
        "--split_group", job["split_group"],
        "--split_name", job["split_name"],
        "--device", device,
        "--seed", str(seed),
        "--limit_rows", str(args.limit_rows),
    ]  # fmt: skip
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    return cmd + (["--overwrite"] if args.overwrite else []) + (["--no_progress"] if args.no_progress else []) + (["--amp"] if args.amp else [])


def main():
    parser = argparse.ArgumentParser(description="Run MPEK with its original settings on every discovered split and seed, one run at a time.")
    add_common_args(parser)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    args.cache_device = args.cache_device or args.device

    jobs = discover_jobs(args)
    maybe_cache(args)
    for job in jobs:
        for seed in args.seeds:
            subprocess.run(train_command(job, seed, args, args.device), check=True, cwd=str(REPO_ROOT))
    print("Run aggregate_mpek_results.py to write the summary tables.")


if __name__ == "__main__":
    main()
