from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from assay_emulator_bench.common import ligand_cache_path, protein_cache_path
from assay_emulator_bench.dataset import CachedMPEKDataset
from assay_emulator_bench.modeling import CachedMPEKRegressor
from assay_emulator_bench.train_single_split import load_checkpoint


def write_cache(cache_root: Path, sequences, smiles_values) -> None:
    for sequence in set(sequences):
        path = protein_cache_path(cache_root, sequence)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embedding=np.arange(4, dtype=np.float32))
    for smiles in set(smiles_values):
        path = ligand_cache_path(cache_root, smiles)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embedding=np.arange(3, dtype=np.float32))


def test_assay_conditions_convert_and_impute_without_dropping_rows(tmp_path):
    frame = pd.DataFrame(
        {
            "sequence": ["AAAA", "AAAA", "BBBB"],
            "smiles": ["C", "C", "CC"],
            "log10_value": [1.0, 2.0, 3.0],
            "pH": [7.0, 60.0, 8.0],
            "Temperature": [298.15, 373.15, 500.0],
            "assay_mapping_status": [
                "matched_expanded",
                "matched_expanded",
                "fallback_collapsed",
            ],
            "pH_imputed": [False, False, True],
            "Temperature_imputed": [False, False, True],
        }
    )
    table_path = tmp_path / "split.parquet"
    cache_root = tmp_path / "cache"
    frame.to_parquet(table_path, index=False)
    write_cache(cache_root, frame["sequence"], frame["smiles"])

    dataset = CachedMPEKDataset(
        str(table_path),
        str(cache_root),
        use_assay_conditions=True,
        temperature_unit="kelvin",
    )

    assert len(dataset) == len(frame)
    assert dataset.condition_diagnostics["pH_range_imputed"] == 1
    assert dataset.condition_diagnostics["Temperature_range_imputed"] == 1
    assert dataset.condition_diagnostics["Temperature_exactly_100C"] == 1
    assert dataset.condition_diagnostics["source_pH_imputed"] == 1
    assert dataset.condition_diagnostics["source_Temperature_imputed"] == 1
    assert dataset[0]["temperature_C"].item() == pytest.approx(25.0)
    assert dataset[1]["pH"].item() == pytest.approx(7.0)
    assert dataset[1]["temperature_C"].item() == pytest.approx(100.0)
    assert dataset[2]["temperature_C"].item() == pytest.approx(25.0)
    assert dataset[1]["pH_range_imputed"].item() is True
    assert dataset[2]["Temperature_range_imputed"].item() is True


def test_no_assay_mode_remains_compatible_with_original_schema(tmp_path):
    frame = pd.DataFrame(
        {
            "sequence": ["AAAA", None],
            "smiles": ["C", "CC"],
            "log10_value": [1.0, 2.0],
        }
    )
    table_path = tmp_path / "split.parquet"
    cache_root = tmp_path / "cache"
    frame.to_parquet(table_path, index=False)
    write_cache(cache_root, frame["sequence"], frame["smiles"])

    dataset = CachedMPEKDataset(str(table_path), str(cache_root))

    assert len(dataset) == 1
    assert dataset.condition_diagnostics["fed_to_model"] is False
    assert dataset.condition_diagnostics["condition_columns_present"] is False
    assert "pH" not in dataset[0]


def test_legacy_checkpoint_defaults_resume_in_no_assay_mode(tmp_path):
    model = CachedMPEKRegressor(
        protein_dim=4,
        ligand_dim=3,
        expert_dim=6,
        num_experts=2,
        tower_hidden=4,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "epoch": 1,
            "best_metric": 0.5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": {},
        },
        checkpoint_path,
    )

    epoch, best_metric = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        torch.device("cpu"),
        SimpleNamespace(use_assay_conditions=False, pair_reducer="none"),
    )

    assert epoch == 1
    assert best_metric == pytest.approx(0.5)


def test_cached_model_supports_plain_and_native_assay_paths():
    torch.manual_seed(7)
    protein = torch.randn(2, 4)
    ligand = torch.randn(2, 3)

    plain = CachedMPEKRegressor(
        protein_dim=4,
        ligand_dim=3,
        expert_dim=6,
        num_experts=2,
        tower_hidden=4,
    ).eval()
    assay = CachedMPEKRegressor(
        protein_dim=4,
        ligand_dim=3,
        expert_dim=6,
        num_experts=2,
        tower_hidden=4,
        use_assay_conditions=True,
        condition_device="cpu",
    ).eval()

    assert plain.ple_input_dim == 6
    assert assay.ple_input_dim == 9
    assert plain(protein, ligand).shape == (2, 1)
    first = assay(
        protein,
        ligand,
        pH=torch.tensor([7.0, 7.0]),
        temperature=torch.tensor([25.0, 25.0]),
    )
    second = assay(
        protein,
        ligand,
        pH=torch.tensor([8.0, 8.0]),
        temperature=torch.tensor([37.0, 37.0]),
    )
    assert first.shape == (2, 1)
    assert not torch.allclose(first, second)


EMULATOR_ROOT = Path(__file__).resolve().parents[4]
ASSAY_ROOT = (
    EMULATOR_ROOT / "data/processed/baselines/MPEK_assayed/kcat"
)


@pytest.mark.skipif(
    not ASSAY_ROOT.exists(),
    reason="Local assay-expanded integration data are unavailable",
)
@pytest.mark.parametrize(
    "relative_path",
    [
        Path("enzyme_sequence_splits/threshold_0.01"),
        Path("uniprot_time_splits"),
    ],
)
def test_local_assay_hierarchy_condition_totals(relative_path):
    totals = {
        "rows": 0,
        "pH_range_imputed": 0,
        "Temperature_range_imputed": 0,
        "Temperature_exactly_100C": 0,
    }
    for partition in ("train", "val", "test"):
        dataset = CachedMPEKDataset(
            str(ASSAY_ROOT / relative_path / f"{partition}.parquet"),
            embeddings_dir="unused",
            use_assay_conditions=True,
            validate_cache=False,
        )
        totals["rows"] += len(dataset)
        for key in totals.keys() - {"rows"}:
            totals[key] += dataset.condition_diagnostics[key]

    assert totals == {
        "rows": 54_536,
        "pH_range_imputed": 11,
        "Temperature_range_imputed": 58,
        "Temperature_exactly_100C": 17,
    }
