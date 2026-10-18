import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (  # noqa: E402,F401
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SEEDS,
    EMULATOR_ROOT,
    MPEK_ROOT,
    discover_split_jobs,
    regression_metrics,
    resolve_amp_dtype,
    run_dir_for,
    save_json,
    set_seed,
    summarize_rows,
)

TASK = "kcat"
DEFAULT_SPLIT_GROUPS = ["enzyme_sequence_splits", "uniprot_time_splits"]
DEFAULT_OUTPUT_DIR = EMULATOR_ROOT / "outputs" / "baselines" / "MPEK_assay"
MODES = ("with_conditions", "without_conditions")
DEFAULT_PH = 7.0
DEFAULT_TEMPERATURE_K = 298.15
PAIR_KEYS = ["sequence", "smiles"]
