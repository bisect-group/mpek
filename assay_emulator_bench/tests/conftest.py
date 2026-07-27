import sys
from pathlib import Path


MPEK_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(MPEK_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MPEK_REPO_ROOT))
