import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from emulator_bench.common import ligand_cache_path, normalize_sequence, protein_cache_path, require_columns


def load_cached_embeddings(keys: pd.Series, cache_path: Callable[[Path, str], Path], embeddings_dir: Path) -> torch.Tensor:
    """Read each unique key once and return one row per input key."""
    codes, uniques = pd.factorize(keys)
    try:
        rows = [np.load(cache_path(embeddings_dir, key))["embedding"].astype(np.float32) for key in uniques]
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
    failed = {item["smiles"] for item in json.loads(failures_path.read_text())["failures"]}
    smiles = frame[smiles_col].astype(str).str.strip()
    drop = smiles.isin(failed) & ~smiles.map({value: ligand_cache_path(embeddings_dir, value).exists() for value in smiles.unique()})
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
        frame, self.rows_dropped = drop_unembedded_ligands(frame, smiles_col, embeddings_dir)
        self.protein = load_cached_embeddings(frame[sequence_col].astype(str).map(normalize_sequence), protein_cache_path, embeddings_dir)
        self.ligand = load_cached_embeddings(frame[smiles_col].astype(str).str.strip(), ligand_cache_path, embeddings_dir)
        raw = frame[target_col].to_numpy(np.float64)
        self.raw_target = torch.from_numpy(raw.astype(np.float32))
        if target_mean is None:
            target_mean, target_std = float(raw.mean()), float(raw.std()) + 1e-5
        self.target_mean, self.target_std = target_mean, target_std
        self.target = torch.from_numpy(((raw - target_mean) / target_std).astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.raw_target)
