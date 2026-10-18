import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MPEK_ROOT = REPO_ROOT / "MTLKcatKM"
MOLE_BERT_ROOT = REPO_ROOT / "Mole-BERT"
EMULATOR_ROOT = REPO_ROOT.parents[1]
DEFAULT_BASE_DIR = EMULATOR_ROOT / "data" / "processed" / "datasets" / "kinetic_dataset"
DEFAULT_OUTPUT_DIR = EMULATOR_ROOT / "outputs" / "baselines" / "MPEK"
DEFAULT_TASKS = ["kcat", "km"]
DEFAULT_SEEDS = [666, 777, 888]
DEFAULT_SPLIT_GROUPS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "substrate_splits",
]
SPLIT_FILES = ("train.parquet", "val.parquet", "test.parquet")

for path in (REPO_ROOT, MPEK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(EMULATOR_ROOT) not in sys.path:
    sys.path.append(str(EMULATOR_ROOT))


def stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split()).replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")


def protein_cache_path(embeddings_dir: Path, sequence: str) -> Path:
    key = stable_hash(normalize_sequence(sequence))
    return Path(embeddings_dir) / "proteins" / key[:2] / f"{key}.npz"


def ligand_cache_path(embeddings_dir: Path, smiles: str) -> Path:
    key = stable_hash(str(smiles).strip())
    return Path(embeddings_dir) / "ligands" / "molebert" / key[:2] / f"{key}.npz"


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def discover_split_jobs(base_dir: Path, tasks=None, split_groups=None) -> List[Dict[str, str]]:
    """Find `<base_dir>/<task>/<split_group>[/threshold_<t>]/{train,val,test}.parquet`."""
    jobs: List[Dict[str, str]] = []
    for task in tasks or DEFAULT_TASKS:
        for split_group in split_groups or DEFAULT_SPLIT_GROUPS:
            group_dir = Path(base_dir) / task / split_group
            if not group_dir.is_dir():
                continue
            split_dirs = [group_dir] + sorted(child for child in group_dir.iterdir() if child.is_dir())
            for split_dir in split_dirs:
                if all((split_dir / name).exists() for name in SPLIT_FILES):
                    jobs.append(
                        {
                            "task": task,
                            "split_group": split_group,
                            "split_name": split_dir.name,
                            "rel_dir": str(split_dir.relative_to(Path(base_dir) / task)),
                            "train_path": str(split_dir / "train.parquet"),
                            "val_path": str(split_dir / "val.parquet"),
                            "test_path": str(split_dir / "test.parquet"),
                        }
                    )
    return jobs


def run_dir_for(output_dir: Path, job: Dict[str, str], seed: int) -> Path:
    return Path(output_dir) / "results" / job["task"] / job["rel_dir"] / f"seed_{seed}"


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan"),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan"),
        "spearman": float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank())) if y_true.size > 1 else float("nan"),
    }


def summarize_rows(rows: List[Dict], group_cols: List[str], metric_cols: Iterable[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = []
    for keys, group in pd.DataFrame(rows).groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["n_runs"] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values):
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)
