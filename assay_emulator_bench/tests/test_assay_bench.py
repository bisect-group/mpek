import numpy as np
import pandas as pd
import pytest
import torch

from assay_emulator_bench.dataset import AssayDataset
from assay_emulator_bench.modeling import CachedMPEKRegressor
from assay_emulator_bench.train_single_split import assay_and_pair_metrics
from emulator_bench.common import ligand_cache_path, protein_cache_path


def write_cache(root, sequences, smiles_values):
    for key, path_fn, size in [(s, protein_cache_path, 4) for s in sequences] + [(s, ligand_cache_path, 3) for s in smiles_values]:
        path = path_fn(root, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embedding=np.arange(size, dtype=np.float32))


@pytest.fixture
def dataset(tmp_path):
    frame = pd.DataFrame(
        {
            "sequence": ["AAAA", "AAAA", "BBBB"],
            "smiles": ["C", "C", "CC"],
            "log10_value": [1.0, 3.0, 2.0],
            "pH": [8.0, None, 7.0],
            "Temperature": [310.15, 298.15, None],
        }
    )
    frame.to_parquet(tmp_path / "split.parquet")
    write_cache(tmp_path / "cache", ["AAAA", "BBBB"], ["C", "CC"])
    return AssayDataset(str(tmp_path / "split.parquet"), str(tmp_path / "cache"))


def test_conditions_are_imputed_and_converted_to_celsius(dataset):
    assert dataset.pH.tolist() == [8.0, 7.0, 7.0]
    assert dataset.temperature.tolist() == pytest.approx([37.0, 25.0, 25.0])
    assert dataset.pair_id.tolist() == [0, 0, 1]


def test_pair_metrics_use_the_maximum_per_pair(dataset):
    y_pred = torch.tensor([1.5, 2.5, 2.25])
    metrics, rows, n_pairs = assay_and_pair_metrics(dataset, y_pred)
    assert n_pairs == 2 and len(rows) == 3
    assert metrics["pair_rmse"] == pytest.approx((((2.5 - 3.0) ** 2 + (2.25 - 2.0) ** 2) / 2) ** 0.5)


def test_condition_input_changes_only_the_conditioned_model():
    protein, ligand = torch.randn(2, 4), torch.randn(2, 3)
    pH, temperature = torch.tensor([7.0, 7.0]), torch.tensor([25.0, 25.0])
    kwargs = dict(device=torch.device("cpu"), expert_dim=6, expert_layers=1, num_experts=2, ple_layers=1, dropout=0.0, tower_layers=2, tower_hidden=4, tower_dropout=0.0, protein_dim=4, ligand_dim=3)
    plain = CachedMPEKRegressor(use_conditions=False, **kwargs).eval()
    conditioned = CachedMPEKRegressor(use_conditions=True, **kwargs).eval()
    assert torch.equal(plain(protein, ligand, pH, temperature), plain(protein, ligand, pH + 1, temperature + 10))
    assert not torch.equal(conditioned(protein, ligand, pH, temperature), conditioned(protein, ligand, pH + 1, temperature + 10))
