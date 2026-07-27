import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assay_emulator_bench.common import MOLE_BERT_ROOT


MOLEBERT_RELATIVE_PATH = Path("model_gin") / "Mole-BERT.pth"
MOLEBERT_RECOVERY_SPEC = "5b6551f:model_gin/Mole-BERT.pth"


def recover_molebert_checkpoint(checkpoint_path: Path) -> bool:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = subprocess.check_output(
            ["git", "show", MOLEBERT_RECOVERY_SPEC],
            cwd=str(MOLE_BERT_ROOT),
        )
    except Exception:
        return False
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
    tmp.replace(checkpoint_path)
    return True


def validate_molebert(auto: bool) -> Path:
    if not MOLE_BERT_ROOT.exists():
        raise FileNotFoundError(f"Mole-BERT submodule directory is missing: {MOLE_BERT_ROOT}")
    checkpoint_path = MOLE_BERT_ROOT / MOLEBERT_RELATIVE_PATH
    if not checkpoint_path.exists() and auto:
        recovered = recover_molebert_checkpoint(checkpoint_path)
        if recovered:
            print(f"Recovered Mole-BERT checkpoint to {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Missing Mole-BERT checkpoint. Expected "
            f"{checkpoint_path}. Run from a submodule checkout that can resolve {MOLEBERT_RECOVERY_SPEC}."
        )
    if checkpoint_path.stat().st_size == 0:
        raise RuntimeError(f"Mole-BERT checkpoint exists but is empty: {checkpoint_path}")
    print(f"Mole-BERT checkpoint OK: {checkpoint_path}")
    return checkpoint_path


def validate_prott5(prottrans_path: str, auto: bool) -> str:
    if Path(prottrans_path).exists():
        print(f"ProtT5 local path OK: {prottrans_path}")
        return prottrans_path
    if prottrans_path != "Rostlab/prot_t5_xl_uniref50":
        raise FileNotFoundError(f"ProtT5 path does not exist: {prottrans_path}")
    if auto:
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            print("huggingface_hub is not importable; transformers will download ProtT5 on first use.")
            return prottrans_path
        snapshot_download(
            repo_id=prottrans_path,
            allow_patterns=["config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json", "special_tokens_map.json"],
        )
        print(f"ProtT5 snapshot available from Hugging Face cache: {prottrans_path}")
    else:
        print(f"ProtT5 will be resolved by transformers from Hugging Face: {prottrans_path}")
    return prottrans_path


def main():
    parser = argparse.ArgumentParser(description="Validate or bootstrap assets needed by the MPEK emulator bench.")
    parser.add_argument("--prottrans_path", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--auto", action="store_true", help="Try to recover/download missing assets.")
    args = parser.parse_args()

    validate_molebert(auto=args.auto)
    validate_prott5(args.prottrans_path, auto=args.auto)


if __name__ == "__main__":
    main()
