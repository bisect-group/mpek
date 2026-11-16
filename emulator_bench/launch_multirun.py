import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import run_dir_for
from emulator_bench.multirun_train import is_complete, train_group
from emulator_bench.run_split_benchmarks import add_common_args, discover_jobs, list_runs, maybe_cache


def main():
    parser = argparse.ArgumentParser(description="Run MPEK with its original settings on every discovered split and seed, many runs of one task together on one GPU.")
    add_common_args(parser)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--group_size", type=int, default=1, help="Runs trained together. Runs of one task share a group, in the order split folder, seed.")
    parser.add_argument("--no_graph", action="store_true", help="Use eager steps instead of CUDA graphs. Results are the same.")
    args = parser.parse_args()
    if args.amp:
        parser.error("--amp is not supported by the multi-run path.")
    args.cache_device = args.cache_device or args.device

    jobs = discover_jobs(args)
    if list_runs(jobs, args):
        return
    maybe_cache(args)
    epochs = args.epochs or 100
    settings = {"epochs": epochs, "limit_rows": args.limit_rows}
    for task in args.tasks:
        runs = []
        for job in (j for j in jobs if j["task"] == task):
            for seed in args.seeds:
                out_dir = run_dir_for(args.output_dir, job, seed)
                if is_complete(out_dir, settings) and not args.overwrite:
                    print(f"Completed run already exists, skipping: {out_dir}")
                    continue
                runs.append({"job": job, "seed": seed, "out_dir": out_dir, "overwrite": args.overwrite or (out_dir / "final_results_test.csv").exists()})
        for start in range(0, len(runs), args.group_size):
            train_group(runs[start : start + args.group_size], str(Path(args.output_dir) / "embeddings"), args.device, epochs, args.limit_rows, not args.no_graph)
    print("Run aggregate_mpek_results.py to write the summary tables.")


if __name__ == "__main__":
    main()
