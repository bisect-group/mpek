import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import MOLE_BERT_ROOT

MOLEBERT_CHECKPOINT = MOLE_BERT_ROOT / "model_gin" / "Mole-BERT.pth"


def validate_molebert() -> Path:
    if not MOLEBERT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing {MOLEBERT_CHECKPOINT}. Run `git submodule update --init Mole-BERT` in the MPEK repository.")
    print(f"Mole-BERT checkpoint OK: {MOLEBERT_CHECKPOINT}")
    return MOLEBERT_CHECKPOINT


def validate_prott5(prottrans_path: str) -> None:
    if Path(prottrans_path).exists():
        print(f"ProtT5 local path OK: {prottrans_path}")
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=prottrans_path,
        allow_patterns=["config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json", "special_tokens_map.json"],
    )
    print(f"ProtT5 snapshot available from the Hugging Face cache: {prottrans_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate the assets needed by the MPEK emulator bench.")
    parser.add_argument("--prottrans_path", type=str, default="Rostlab/prot_t5_xl_uniref50")
    args = parser.parse_args()
    validate_molebert()
    validate_prott5(args.prottrans_path)


if __name__ == "__main__":
    main()
