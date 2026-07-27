import pandas as pd
import pytest

from assay_emulator_bench.evaluation import (
    aggregate_pair_predictions,
    checkpoint_rmse,
    evaluate_prediction_rows,
)


def test_max_pair_aggregation_uses_one_vote_per_pair():
    assay_rows = pd.DataFrame(
        {
            "sequence": ["A", "A", "B"],
            "smiles": ["C", "C", "CC"],
            "y_true": [1.0, 3.0, 2.0],
            "y_pred": [1.5, 2.5, 2.25],
        }
    )

    pairs = aggregate_pair_predictions(assay_rows, "max")

    assert pairs.to_dict("records") == [
        {
            "sequence": "A",
            "smiles": "C",
            "n_assays": 2,
            "y_true": 3.0,
            "y_pred": 2.5,
        },
        {
            "sequence": "B",
            "smiles": "CC",
            "n_assays": 1,
            "y_true": 2.0,
            "y_pred": 2.25,
        },
    ]


def test_pair_rmse_selects_checkpoint_when_pair_reduction_is_enabled():
    assay_rows = pd.DataFrame(
        {
            "sequence": ["A", "A", "B"],
            "smiles": ["C", "C", "CC"],
            "y_true": [1.0, 3.0, 2.0],
            "y_pred": [1.5, 2.5, 2.25],
        }
    )
    row_metrics, pairs, pair_metrics = evaluate_prediction_rows(assay_rows, "max")

    assert len(pairs) == 2
    assert checkpoint_rmse(row_metrics, pair_metrics, "max") == pytest.approx(
        pair_metrics["rmse"]
    )
    assert checkpoint_rmse(row_metrics, None, "none") == pytest.approx(
        row_metrics["rmse"]
    )
