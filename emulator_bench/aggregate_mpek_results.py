import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from emulator_bench.common import DEFAULT_OUTPUT_DIR, summarize_rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate MPEK test metrics over seeds.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    files = sorted((output_dir / "results").glob("*/**/seed_*/final_results_test.csv"))
    if not files:
        raise FileNotFoundError(f"No final_results_test.csv under {output_dir / 'results'}")
    runs = pd.concat([pd.read_csv(path).assign(run_dir=str(path.parent)) for path in files], ignore_index=True)
    runs.to_csv(output_dir / "mpek_summary_runs.csv", index=False)
    metrics = ["mse", "rmse", "mae", "r2", "pearson", "spearman"]
    summarize_rows(runs.to_dict("records"), ["task", "split_group", "split_name"], metrics).to_csv(output_dir / "mpek_summary.csv", index=False)
    print(f"Saved {output_dir / 'mpek_summary_runs.csv'} and {output_dir / 'mpek_summary.csv'}")


if __name__ == "__main__":
    main()
