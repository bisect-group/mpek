from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from assay_emulator_bench.common import DEFAULT_PH, DEFAULT_TEMPERATURE_K, PAIR_KEYS
from emulator_bench.common import ligand_cache_path, normalize_sequence, protein_cache_path, require_columns
from emulator_bench.dataset import drop_unembedded_ligands, load_cached_embeddings

TARGET_COL = "log10_value"


class AssayDataset:
    """One row per assay: cached ProtT5 and Mole-BERT vectors, pH, temperature in Celsius, train-normalised target."""

    def __init__(
        self,
        table_path: str,
        embeddings_dir: str,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        limit_rows: int = 0,
    ):
        frame = pd.read_parquet(table_path)
        require_columns(frame, [*PAIR_KEYS, TARGET_COL, "pH", "Temperature"], Path(table_path))
        if limit_rows:
            frame = frame.head(limit_rows)
        frame, self.rows_dropped = drop_unembedded_ligands(frame.dropna(subset=[*PAIR_KEYS, TARGET_COL]), "smiles", embeddings_dir)
        self.pair_id = torch.from_numpy(frame.groupby(PAIR_KEYS, sort=False).ngroup().to_numpy())
        self.protein = load_cached_embeddings(frame["sequence"].astype(str).map(normalize_sequence), protein_cache_path, embeddings_dir)
        self.ligand = load_cached_embeddings(frame["smiles"].astype(str).str.strip(), ligand_cache_path, embeddings_dir)
        self.pH = torch.from_numpy(frame["pH"].fillna(DEFAULT_PH).to_numpy("float32"))
        self.temperature = torch.from_numpy((frame["Temperature"].fillna(DEFAULT_TEMPERATURE_K) - 273.15).to_numpy("float32"))
        self.raw_target = torch.from_numpy(frame[TARGET_COL].to_numpy("float32"))
        if target_mean is None:
            target_mean, target_std = float(self.raw_target.mean()), float(self.raw_target.std(unbiased=False)) + 1e-5
        self.target_mean, self.target_std = target_mean, target_std
        self.target = ((self.raw_target - target_mean) / target_std).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.raw_target)
