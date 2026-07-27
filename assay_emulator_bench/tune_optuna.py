import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assay_emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_SPLIT_GROUPS, DEFAULT_TASKS, DEFAULT_SEEDS, discover_split_jobs, normalize_threshold_args
from assay_emulator_bench.run_split_benchmarks import maybe_cache, train_command


def metric_direction(metric: str) -> str:
    return "minimize" if metric in {"rmse", "mse", "mae", "loss"} else "maximize"


def sqlite_path_from_storage(storage: str):
    if not storage or not storage.startswith("sqlite:///"):
        return None
    parsed = urlparse(storage)
    return Path(unquote(parsed.path)) if parsed.path else None


def prepare_storage(args):
    db_path = sqlite_path_from_storage(args.storage)
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and args.reset_storage:
        db_path.unlink()
    if db_path.exists():
        with sqlite3.connect(str(db_path)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if tables and "version_info" not in tables:
            raise RuntimeError(f"Existing SQLite file is not an Optuna DB: {db_path}")


def suggest_hparams(trial, args):
    return {
        "batch_size": args.batch_size or trial.suggest_categorical("batch_size", [1, 2, 4, 8]),
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True),
        "clip_grad": trial.suggest_categorical("clip_grad", [0.5, 1.0, 2.0, 5.0]),
    }


def run_trial_job(job, seed, hparams, args, trial_number):
    trial_root = Path(job["root_dir"]) / "mpek_optuna_runs" / f"trial_{trial_number}" / f"seed_{seed}"
    if not (trial_root / f"final_results_{args.eval_split}.csv").exists() or args.overwrite_runs:
        args.overwrite = args.overwrite_runs
        old_values = {key: getattr(args, key) for key in hparams}
        for key, value in hparams.items():
            setattr(args, key, value)
        cmd = train_command(job, seed, trial_root, args, args.device)
        for key, value in old_values.items():
            setattr(args, key, value)
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    metrics = pd.read_csv(trial_root / f"final_results_{args.eval_split}.csv").iloc[0].to_dict()
    if args.metric not in metrics:
        raise RuntimeError(f"Metric `{args.metric}` not found in {trial_root}")
    return float(metrics[args.metric])


def main():
    parser = argparse.ArgumentParser(description="Tune optimization-only MPEK retraining hyperparameters with Optuna.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=None)
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
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mse", "mae", "r2", "pearson", "spearman", "loss"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="mpek_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--reset_storage", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    if args.storage is None:
        args.storage = "sqlite:///%s" % (Path(args.base_dir) / "optuna_studies" / f"{args.study_name}.db")
    maybe_cache(args)
    prepare_storage(args)
    jobs = discover_split_jobs(Path(args.base_dir), tasks=args.tasks, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    study = optuna.create_study(
        direction=metric_direction(args.metric),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    def objective(trial):
        hparams = suggest_hparams(trial, args)
        scores = []
        for job in jobs:
            for seed in args.seeds:
                scores.append(run_trial_job(job, seed, hparams, args, trial.number))
        return float(sum(scores) / len(scores))

    study.optimize(objective, n_trials=args.n_trials)
    out_dir = Path(args.base_dir) / "optuna_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_dir / f"{args.study_name}_trials.csv", index=False)
    with open(out_dir / f"{args.study_name}_best_hparams.json", "w") as handle:
        json.dump(
            {
                "study_name": args.study_name,
                "storage": args.storage,
                "best_trial_number": int(study.best_trial.number),
                "best_value": float(study.best_value),
                "best_hparams": dict(study.best_params),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
