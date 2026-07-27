from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from assay_emulator_bench.common import ligand_cache_path, normalize_sequence, protein_cache_path, read_table, require_columns


DEFAULT_PH = 7.0
DEFAULT_TEMPERATURE_C = 25.0
KELVIN_OFFSET = 273.15


class CachedMPEKDataset(Dataset):
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
        validate_cache: bool = True,
        preload: bool = False,
        use_assay_conditions: bool = False,
        ph_col: str = "pH",
        temperature_col: str = "Temperature",
        temperature_unit: str = "kelvin",
    ):
        self.table_path = Path(table_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.use_assay_conditions = bool(use_assay_conditions)
        self.ph_col = ph_col
        self.temperature_col = temperature_col
        self.temperature_unit = temperature_unit
        if temperature_unit not in {"kelvin", "celsius"}:
            raise ValueError(f"Unsupported temperature unit: {temperature_unit}")

        frame = read_table(self.table_path)
        if limit_rows:
            frame = frame.head(limit_rows)
        required = [sequence_col, smiles_col, target_col]
        if self.use_assay_conditions:
            required.extend([ph_col, temperature_col])
        require_columns(frame, required, self.table_path)

        if self.use_assay_conditions:
            missing_required = frame[required].isna().any()
            missing_columns = missing_required[missing_required].index.tolist()
            if missing_columns:
                raise ValueError(
                    f"Required columns contain null values in {self.table_path}: "
                    f"{missing_columns}"
                )
        else:
            frame = frame.dropna(
                subset=[sequence_col, smiles_col, target_col]
            )
        frame = frame.reset_index(drop=True)
        self.row_indices = np.arange(len(frame), dtype=np.int64)
        self.sequences = [normalize_sequence(value) for value in frame[sequence_col].astype(str).tolist()]
        self.smiles = [str(value).strip() for value in frame[smiles_col].astype(str).tolist()]
        self.targets = frame[target_col].astype(float).to_numpy(dtype=np.float32)
        if not np.isfinite(self.targets).all():
            raise ValueError(f"Non-finite targets in {self.table_path}")

        self.assay_mapping_status = (
            frame["assay_mapping_status"].fillna("unknown").astype(str).tolist()
            if "assay_mapping_status" in frame
            else ["unknown"] * len(frame)
        )
        self.source_ph_imputed = (
            frame["pH_imputed"].fillna(False).astype(bool).to_numpy()
            if "pH_imputed" in frame
            else np.zeros(len(frame), dtype=bool)
        )
        self.source_temperature_imputed = (
            frame["Temperature_imputed"].fillna(False).astype(bool).to_numpy()
            if "Temperature_imputed" in frame
            else np.zeros(len(frame), dtype=bool)
        )
        self.has_condition_metadata = ph_col in frame and temperature_col in frame
        self.ph_values = None
        self.temperature_c_values = None
        self.ph_range_imputed = np.zeros(len(frame), dtype=bool)
        self.temperature_range_imputed = np.zeros(len(frame), dtype=bool)
        self.condition_diagnostics = self._prepare_conditions(frame)

        self.target_mean = float(np.mean(self.targets)) if target_mean is None else float(target_mean)
        std = float(np.std(self.targets)) if target_std is None else float(target_std)
        self.target_std = std if std > 1e-8 else 1.0
        self._preloaded = None

        if validate_cache:
            self._validate_cache()
        if preload:
            self._preloaded = [self._load_features(index) for index in range(len(self))]

    @staticmethod
    def _finite_range(values: np.ndarray) -> Dict[str, Optional[float]]:
        finite = values[np.isfinite(values)]
        return {
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
        }

    def _prepare_conditions(self, frame) -> Dict:
        diagnostics = {
            "table_path": str(self.table_path),
            "fed_to_model": self.use_assay_conditions,
            "rows": int(len(frame)),
            "condition_columns_present": self.has_condition_metadata,
            "temperature_input_unit": self.temperature_unit,
            "source_pH_imputed": int(self.source_ph_imputed.sum()),
            "source_Temperature_imputed": int(self.source_temperature_imputed.sum()),
        }
        if not self.has_condition_metadata:
            if self.use_assay_conditions:
                raise ValueError(
                    f"Assay conditions requested but {self.ph_col} and "
                    f"{self.temperature_col} are not both present in {self.table_path}"
                )
            diagnostics.update(
                {
                    "pH_range_imputed": 0,
                    "Temperature_range_imputed": 0,
                    "Temperature_exactly_100C": 0,
                }
            )
            return diagnostics

        raw_ph = frame[self.ph_col].astype(float).to_numpy(dtype=np.float64)
        raw_temperature = frame[self.temperature_col].astype(float).to_numpy(dtype=np.float64)
        if not np.isfinite(raw_ph).all() or not np.isfinite(raw_temperature).all():
            raise ValueError(f"Non-finite assay conditions in {self.table_path}")
        temperature_c = (
            raw_temperature - KELVIN_OFFSET
            if self.temperature_unit == "kelvin"
            else raw_temperature.copy()
        )

        self.ph_range_imputed = (raw_ph < 0.0) | (raw_ph >= 14.0)
        self.temperature_range_imputed = (temperature_c < 0.0) | (temperature_c > 100.0)
        final_ph = raw_ph.copy()
        final_temperature_c = temperature_c.copy()
        final_ph[self.ph_range_imputed] = DEFAULT_PH
        final_temperature_c[self.temperature_range_imputed] = DEFAULT_TEMPERATURE_C
        self.ph_values = final_ph.astype(np.float32)
        self.temperature_c_values = final_temperature_c.astype(np.float32)

        diagnostics.update(
            {
                "raw_pH": self._finite_range(raw_ph),
                "raw_Temperature": self._finite_range(raw_temperature),
                "raw_Temperature_C": self._finite_range(temperature_c),
                "final_pH": self._finite_range(final_ph),
                "final_Temperature_C": self._finite_range(final_temperature_c),
                "pH_range_imputed": int(self.ph_range_imputed.sum()),
                "Temperature_range_imputed": int(self.temperature_range_imputed.sum()),
                "Temperature_exactly_100C": int(
                    np.isclose(temperature_c, 100.0, atol=1e-6).sum()
                ),
            }
        )
        return diagnostics

    def _validate_cache(self) -> None:
        missing = []
        for sequence, smiles in zip(self.sequences, self.smiles):
            p_path = protein_cache_path(self.embeddings_dir, sequence)
            l_path = ligand_cache_path(self.embeddings_dir, smiles)
            if not p_path.exists():
                missing.append(str(p_path))
            if not l_path.exists():
                missing.append(str(l_path))
            if len(missing) >= 10:
                break
        if missing:
            raise FileNotFoundError("Missing cached embeddings. Run cache_embeddings.py first. Examples: " + "; ".join(missing))

    def _load_features(self, index: int):
        protein = np.load(protein_cache_path(self.embeddings_dir, self.sequences[index]), allow_pickle=True)["embedding"].astype(np.float32)
        ligand = np.load(ligand_cache_path(self.embeddings_dir, self.smiles[index]), allow_pickle=True)["embedding"].astype(np.float32)
        return protein, ligand

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        if self._preloaded is None:
            protein, ligand = self._load_features(index)
        else:
            protein, ligand = self._preloaded[index]
        raw_target = float(self.targets[index])
        norm_target = (raw_target - self.target_mean) / self.target_std
        sample = {
            "protein": torch.from_numpy(protein),
            "ligand": torch.from_numpy(ligand),
            "target": torch.tensor([norm_target], dtype=torch.float32),
            "raw_target": torch.tensor([raw_target], dtype=torch.float32),
            "row_index": torch.tensor(self.row_indices[index], dtype=torch.int64),
            "sequence": self.sequences[index],
            "smiles": self.smiles[index],
            "assay_mapping_status": self.assay_mapping_status[index],
            "source_pH_imputed": torch.tensor(
                self.source_ph_imputed[index], dtype=torch.bool
            ),
            "source_Temperature_imputed": torch.tensor(
                self.source_temperature_imputed[index], dtype=torch.bool
            ),
            "pH_range_imputed": torch.tensor(
                self.ph_range_imputed[index], dtype=torch.bool
            ),
            "Temperature_range_imputed": torch.tensor(
                self.temperature_range_imputed[index], dtype=torch.bool
            ),
        }
        if self.has_condition_metadata:
            sample["pH"] = torch.tensor(self.ph_values[index], dtype=torch.float32)
            sample["temperature_C"] = torch.tensor(
                self.temperature_c_values[index], dtype=torch.float32
            )
        return sample
