import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assay_emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_RESULTS_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_TASKS,
    DEFAULT_SEEDS,
    discover_split_jobs,
    normalize_threshold_args,
    split_sizes,
)

CACHE_SCRIPT = REPO_ROOT / "assay_emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "assay_emulator_bench" / "train_single_split.py"


def maybe_cache(args):
    if args.skip_cache:
        return
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
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
        "--device",
        args.cache_device,
        "--prottrans_path",
        args.prottrans_path,
        "--protein_batch_size",
        str(args.protein_batch_size),
        "--ligand_batch_size",
        str(args.ligand_batch_size),
        "--protein_dtype",
        args.protein_dtype,
    ]
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.cache_overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def train_command(job, seed, run_dir, args, device):
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_path",
        job["train_path"],
        "--val_path",
        job["val_path"],
        "--test_path",
        job["test_path"],
        "--embeddings_dir",
        args.embeddings_dir,
        "--out_dir",
        str(run_dir),
        "--task",
        job["task"],
        "--split_group",
        job["split_group"],
        "--split_name",
        job["split_name"],
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
        device,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--expert_dim",
        str(args.expert_dim),
        "--expert_layers",
        str(args.expert_layers),
        "--num_experts",
        str(args.num_experts),
        "--ple_layers",
        str(args.ple_layers),
        "--dropout",
        str(args.dropout),
        "--tower_layers",
        str(args.tower_layers),
        "--tower_hidden",
        str(args.tower_hidden),
        "--tower_dropout",
        str(args.tower_dropout),
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--clip_grad",
        str(args.clip_grad),
    ] + (["--use_assay_conditions"] if args.use_assay_conditions else []) + (["--pin_memory"] if args.pin_memory else []) + (["--persistent_workers"] if args.persistent_workers else []) + (["--preload"] if args.preload else []) + (["--overwrite"] if args.overwrite else []) + (["--no_progress"] if args.no_progress else [])


def main():
    parser = argparse.ArgumentParser(description="Run MPEK cached retraining across discovered EMULaToR split jobs.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
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

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache(args)
    jobs = discover_split_jobs(Path(args.base_dir), tasks=args.tasks, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    rows = []
    for job in jobs:
        for seed in args.seeds:
            run_dir = Path(job["root_dir"]) / args.results_dirname / f"seed_{seed}"
            subprocess.run(train_command(job, seed, run_dir, args, args.device), check=True, cwd=str(REPO_ROOT))
            metrics_path = run_dir / "final_results_test.csv"
            metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
            row = {"task": job["task"], "split_group": job["split_group"], "split_name": job["split_name"], "difficulty": job["difficulty"], "seed": seed, "run_dir": str(run_dir)}
            row.update(split_sizes(job))
            row.update({f"test_{key}": value for key, value in metrics.items()})
            assay_row_metrics_path = (
                run_dir / "final_results_test_assay_rows.csv"
            )
            if assay_row_metrics_path.exists():
                assay_row_metrics = pd.read_csv(assay_row_metrics_path).iloc[0].to_dict()
                row.update(
                    {
                        f"test_assay_row_{key}": value
                        for key, value in assay_row_metrics.items()
                    }
                )
            rows.append(row)
    out_path = Path(args.base_dir) / "mpek_summary_runs.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
