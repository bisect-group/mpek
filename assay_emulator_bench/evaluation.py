from typing import Dict, Tuple

import pandas as pd

from assay_emulator_bench.common import regression_metrics


PAIR_KEYS = ["sequence", "smiles"]


def aggregate_pair_predictions(
    assay_rows: pd.DataFrame,
    reducer: str,
) -> pd.DataFrame:
    if reducer != "max":
        raise ValueError(f"Unsupported pair reducer: {reducer}")
    required = [*PAIR_KEYS, "y_true", "y_pred"]
    missing = [column for column in required if column not in assay_rows]
    if missing:
        raise ValueError(f"Prediction rows are missing required columns: {missing}")
    if assay_rows[required].isna().any().any():
        raise ValueError("Prediction rows contain null pair identifiers or values")

    pairs = (
        assay_rows.groupby(PAIR_KEYS, sort=False, as_index=False)
        .agg(
            n_assays=("y_true", "size"),
            y_true=("y_true", "max"),
            y_pred=("y_pred", "max"),
        )
        .reset_index(drop=True)
    )
    if len(pairs) != assay_rows[PAIR_KEYS].drop_duplicates().shape[0]:
        raise RuntimeError("Pair aggregation did not produce one row per pair")
    return pairs


def evaluate_prediction_rows(
    assay_rows: pd.DataFrame,
    pair_reducer: str,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float] | None]:
    row_metrics = regression_metrics(assay_rows["y_true"], assay_rows["y_pred"])
    if pair_reducer == "none":
        return row_metrics, pd.DataFrame(), None
    pairs = aggregate_pair_predictions(assay_rows, pair_reducer)
    pair_metrics = regression_metrics(pairs["y_true"], pairs["y_pred"])
    return row_metrics, pairs, pair_metrics


def checkpoint_rmse(
    row_metrics: Dict[str, float],
    pair_metrics: Dict[str, float] | None,
    pair_reducer: str,
) -> float:
    if pair_reducer == "none":
        return float(row_metrics["rmse"])
    if pair_metrics is None:
        raise ValueError("Pair metrics are required when pair reduction is enabled")
    return float(pair_metrics["rmse"])
