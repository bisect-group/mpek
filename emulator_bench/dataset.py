import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from emulator_bench.common import ligand_cache_path, protein_cache_path, protein_text, require_columns


def load_cached_embeddings(keys: pd.Series, cache_path: Callable[[Path, str], Path], embeddings_dir: Path) -> torch.Tensor:
    """Read each unique key once and return one row per input key."""
    codes, uniques = pd.factorize(keys)
    try:
        rows = [np.load(cache_path(embeddings_dir, key)) for key in uniques]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing cached embedding. Run cache_embeddings.py first. {exc}") from exc
    return torch.from_numpy(np.stack(rows)[codes])


def drop_unembedded_ligands(frame: pd.DataFrame, smiles_col: str, embeddings_dir: str) -> tuple[pd.DataFrame, int]:
    """Drop rows whose ligand the caching step could not featurize (listed in ligand_cache_failures.json).

    Other missing embeddings stay and raise in load_cached_embeddings.
    """
    failures_path = Path(embeddings_dir) / "ligand_cache_failures.json"
    if not failures_path.exists():
        return frame, 0
    drop = frame[smiles_col].isin({item["smiles"] for item in json.loads(failures_path.read_text())["failures"]})
    return frame[~drop].reset_index(drop=True), int(drop.sum())


class CachedMPEKDataset:
    """Cached ProtT5 and Mole-BERT features with train-normalised targets for one split file."""

    def __init__(
        self,
        table_path: str,
        embeddings_dir: str,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: str = "log10_value",
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        limit_rows: int = 0,
    ):
        frame = pd.read_parquet(table_path)
        require_columns(frame, [sequence_col, smiles_col, target_col], Path(table_path))
        if limit_rows:
            frame = frame.head(limit_rows)
        frame = frame[[sequence_col, smiles_col, target_col]].dropna()
        frame[[sequence_col, smiles_col]] = frame[[sequence_col, smiles_col]].astype(str)
        frame, self.rows_dropped = drop_unembedded_ligands(frame, smiles_col, embeddings_dir)
        self.protein = load_cached_embeddings(frame[sequence_col].map(protein_text), protein_cache_path, embeddings_dir)
        self.ligand = load_cached_embeddings(frame[smiles_col], ligand_cache_path, embeddings_dir)
        raw = frame[target_col].to_numpy(np.float64)
        labels = raw.astype(np.float32)  # the original reads labels as float32 but takes mean and std from the float64 file values
        if target_mean is None:
            target_mean, target_std = float(raw.mean()), float(raw.std()) + 1e-5
        self.target_mean, self.target_std = target_mean, target_std
        self.raw_target = torch.from_numpy(labels)
        self.target = torch.from_numpy(((labels.astype(np.float64) - target_mean) / target_std).astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.raw_target)
