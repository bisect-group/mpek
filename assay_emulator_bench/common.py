import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MPEK_ROOT = REPO_ROOT / "MTLKcatKM"
MOLE_BERT_ROOT = REPO_ROOT / "Mole-BERT"
DEFAULT_BASE_DIR = Path(
    "/home/adhil/github/EMULaToR/data/processed/baselines/MPEK_assayed"
)
DEFAULT_EMBEDDINGS_DIR = Path(
    "/home/adhil/github/EMULaToR/data/processed/baselines/MPEK/embeddings"
)
DEFAULT_RESULTS_DIRNAME = "mpek_results"
DEFAULT_TASKS = ["kcat"]
DEFAULT_SEEDS = [666, 777, 888]
DEFAULT_SPLIT_GROUPS = [
    "enzyme_sequence_splits",
    "uniprot_time_splits",
]

for path in (REPO_ROOT, MPEK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


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


def load_json(path: Path) -> Dict:
    with open(path, "r") as handle:
        return json.load(handle)


def append_csv_row(path: Path, row: Dict) -> None:
    ensure_parent(path)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_fast_math() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def resolve_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None, "fp32"
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def read_table(path: Path) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def _find_split_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv", ".tsv", ".txt"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _threshold_value(name: str) -> float:
    try:
        return float(name.split("threshold_")[-1])
    except Exception:
        return math.inf


def _difficulty_labels(names: List[str]) -> Dict[str, str]:
    ordered = sorted(names, key=_threshold_value)
    if len(ordered) == 1:
        return {ordered[0]: "single"}
    if len(ordered) == 2:
        return {ordered[0]: "hard", ordered[1]: "easy"}
    if len(ordered) == 3:
        return {ordered[0]: "hard", ordered[1]: "medium", ordered[2]: "easy"}
    return {name: f"rank_{idx}" for idx, name in enumerate(ordered, 1)}


def normalize_threshold_args(thresholds=None, threshold=None) -> Optional[List[str]]:
    values = []
    if thresholds:
        values.extend(str(value) for value in thresholds if str(value).strip())
    if threshold and str(threshold).strip():
        values.append(str(threshold))
    if not values:
        return None
    out, seen = [], set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def discover_split_jobs(base_dir: Path, tasks=None, split_groups=None, thresholds=None) -> List[Dict[str, str]]:
    base_dir = Path(base_dir)
    tasks = list(tasks or DEFAULT_TASKS)
    split_groups = list(split_groups or DEFAULT_SPLIT_GROUPS)
    threshold_filter = set(thresholds) if thresholds is not None else None
    jobs: List[Dict[str, str]] = []

    for task in tasks:
        task_dir = base_dir / task
        if not task_dir.exists():
            continue
        for split_group in split_groups:
            group_dir = task_dir / split_group
            if not group_dir.exists():
                continue

            train_path = _find_split_file(group_dir, "train")
            val_path = _find_split_file(group_dir, "val")
            test_path = _find_split_file(group_dir, "test")
            if train_path and val_path and test_path:
                jobs.append(
                    {
                        "task": task,
                        "split_group": split_group,
                        "split_name": split_group,
                        "difficulty": split_group,
                        "root_dir": str(group_dir),
                        "train_path": str(train_path),
                        "val_path": str(val_path),
                        "test_path": str(test_path),
                    }
                )
                continue

            children = [child for child in sorted(group_dir.iterdir()) if child.is_dir()]
            if threshold_filter is not None:
                children = [child for child in children if child.name in threshold_filter]
            threshold_names = [child.name for child in children if child.name.startswith("threshold_")]
            difficulty_by_name = _difficulty_labels(threshold_names)
            for child in children:
                train_path = _find_split_file(child, "train")
                val_path = _find_split_file(child, "val")
                test_path = _find_split_file(child, "test")
                if not (train_path and val_path and test_path):
                    continue
                jobs.append(
                    {
                        "task": task,
                        "split_group": split_group,
                        "split_name": child.name,
                        "difficulty": difficulty_by_name.get(child.name, child.name),
                        "root_dir": str(child),
                        "train_path": str(train_path),
                        "val_path": str(val_path),
                        "test_path": str(test_path),
                    }
                )
    return jobs


def split_sizes(job: Dict[str, str]) -> Dict[str, int]:
    return {
        "train_size": len(read_table(Path(job["train_path"]))),
        "val_size": len(read_table(Path(job["val_path"]))),
        "test_size": len(read_table(Path(job["test_path"]))),
    }


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan")
    spearman = float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank())) if y_true.size > 1 else float("nan")
    return {"mse": mse, "rmse": mse ** 0.5, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


def summarize_rows(rows: List[Dict], group_cols: Iterable[str], metric_cols: Iterable[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    for keys, group in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(group))
        for metric in metric_cols:
            if metric in group:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if len(values):
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_var"] = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)
