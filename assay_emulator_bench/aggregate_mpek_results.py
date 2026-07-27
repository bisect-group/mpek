import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assay_emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_RESULTS_DIRNAME, DEFAULT_SPLIT_GROUPS, DEFAULT_TASKS, discover_split_jobs, normalize_threshold_args, split_sizes, summarize_rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate MPEK emulator bench result CSVs.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    args = parser.parse_args()
    thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    jobs = discover_split_jobs(Path(args.base_dir), tasks=args.tasks, split_groups=args.split_groups, thresholds=thresholds)
    rows = []
    for job in jobs:
        results_root = Path(job["root_dir"]) / args.results_dirname
        for metrics_path in sorted(results_root.glob("seed_*/final_results_test.csv")):
            metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
            row = {"task": job["task"], "split_group": job["split_group"], "split_name": job["split_name"], "difficulty": job["difficulty"], "seed": metrics_path.parent.name.replace("seed_", ""), "run_dir": str(metrics_path.parent)}
            row.update(split_sizes(job))
            row.update({f"test_{key}": value for key, value in metrics.items()})
            assay_row_path = metrics_path.parent / "final_results_test_assay_rows.csv"
            if assay_row_path.exists():
                assay_row_metrics = pd.read_csv(assay_row_path).iloc[0].to_dict()
                row.update(
                    {
                        f"test_assay_row_{key}": value
                        for key, value in assay_row_metrics.items()
                    }
                )
            rows.append(row)
    runs = pd.DataFrame(rows)
    runs_path = Path(args.base_dir) / "mpek_summary_runs.csv"
    runs.to_csv(runs_path, index=False)
    metric_cols = [col for col in runs.columns if col.startswith("test_") and pd.api.types.is_numeric_dtype(runs[col])] if len(runs) else []
    summarize_rows(rows, ["task", "split_group", "split_name", "difficulty"], metric_cols).to_csv(Path(args.base_dir) / "mpek_summary_thresholds.csv", index=False)
    summarize_rows(rows, ["task", "split_group"], metric_cols).to_csv(Path(args.base_dir) / "mpek_summary_by_split_group.csv", index=False)
    print(f"Saved {runs_path}")


if __name__ == "__main__":
    main()
