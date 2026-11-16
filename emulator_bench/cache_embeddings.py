import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from emulator_bench.common import (
    CACHE_VERSION,
    CATAPRO_MAX_RESIDUES,
    CATAPRO_PROTEIN_CACHE_DIR,
    DEFAULT_BASE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_TASKS,
    MOLE_BERT_ROOT,
    PROTEIN_MAX_LENGTH,
    PROTEIN_MODEL_ID,
    catapro_protein_path,
    discover_split_jobs,
    ligand_cache_path,
    protein_cache_path,
    protein_text,
    require_columns,
    save_json,
    save_npy_atomic,
)
from emulator_bench.prepare_assets import validate_molebert
from src.utils.rich_progress import progress

FAILURES_NAME = "ligand_cache_failures.json"


def collect_unique_values(jobs, sequence_col: str, smiles_col: str, target_col: str, limit_rows: int = 0):
    sequences, smiles_values = set(), set()
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            path = Path(job[split_key])
            frame = pd.read_parquet(path)
            require_columns(frame, [sequence_col, smiles_col, target_col], path)
            if limit_rows:
                frame = frame.head(limit_rows)
            frame = frame[[sequence_col, smiles_col, target_col]].dropna()
            sequences.update(frame[sequence_col].astype(str))
            smiles_values.update(frame[smiles_col].astype(str))
    return sorted(sequences), sorted(smiles_values)


def protein_inputs(tokenizer, text: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids and mask as MTLKDataset builds them: truncation and padding to 1024, batch of one."""
    ids = tokenizer(text, truncation=True, padding="max_length", max_length=PROTEIN_MAX_LENGTH)
    return torch.tensor([ids["input_ids"]], device=device), torch.tensor([ids["attention_mask"]], device=device)


def load_protein_encoder(prottrans_path: str, device: torch.device):
    from transformers import T5Tokenizer

    from MTLKcatKM.encoder.protein_encoder import ProteinEncoder

    tokenizer = T5Tokenizer.from_pretrained(prottrans_path, do_lower_case=False, legacy=False)
    return tokenizer, ProteinEncoder(model_dir=prottrans_path, device=device, frozen_params=True).to(device)


def embed_proteins(args, sequences):
    """Copy CataPro vectors where the ProtT5 input is identical, compute the others with the original ProteinEncoder."""
    texts = sorted({protein_text(seq): len(seq) for seq in sequences}.items())
    missing = [(text, n) for text, n in texts if args.overwrite or not protein_cache_path(args.embeddings_dir, text).exists()]
    reused, compute = 0, []
    for text, n in missing:
        source = catapro_protein_path(text)
        if not args.no_catapro and n <= CATAPRO_MAX_RESIDUES and source.exists():
            save_npy_atomic(protein_cache_path(args.embeddings_dir, text), np.load(source))
            reused += 1
        else:
            compute.append(text)
    if compute:
        device = torch.device(args.device)
        print(f"Computing {len(compute)} ProtT5 vectors on {device}")
        tokenizer, encoder = load_protein_encoder(args.prottrans_path, device)
        for text in progress(compute, desc="ProtT5", unit="sequence"):
            vector = encoder(*protein_inputs(tokenizer, text, device))[0]
            save_npy_atomic(protein_cache_path(args.embeddings_dir, text), vector.float().cpu().numpy())
    return {"total": len(texts), "present": len(texts) - len(missing), "copied_from_catapro": reused, "computed": len(compute)}


def load_molebert(device: torch.device):
    validate_molebert()
    from Mole_BERT.model import GNN

    model = GNN(5, 300, JK="last", drop_ratio=0.0, gnn_type="gin")
    model.load_state_dict(torch.load(MOLE_BERT_ROOT / "model_gin" / "Mole-BERT.pth", map_location=device))
    return model.to(device).eval()


def featurize(smiles: str):
    """Graph of the original loader (MTLKDataset.__getitem__); raises where the original raises."""
    from Mole_BERT.loader import mol_to_graph_data_obj_simple

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("rdkit_parse_failed")
    return mol_to_graph_data_obj_simple(mol)


@torch.no_grad()
def molebert_vectors(model, graphs, device) -> np.ndarray:
    """Mean over the node vectors of each graph (pool 'mean' of the frozen encoder in eval mode)."""
    batch = Batch.from_data_list(graphs).to(device)
    return global_mean_pool(model(batch.x, batch.edge_index, batch.edge_attr), batch.batch).float().cpu().numpy()


def embed_ligands(args, smiles_values):
    failures_path = args.embeddings_dir / FAILURES_NAME
    failures = {item["smiles"]: item["error"] for item in json.loads(failures_path.read_text())["failures"]} if failures_path.exists() else {}
    pending = [s for s in smiles_values if args.overwrite or not (ligand_cache_path(args.embeddings_dir, s).exists() or s in failures)]
    written = 0
    if pending:
        device = torch.device(args.device)
        model = load_molebert(device)
        for start in progress(range(0, len(pending), args.ligand_batch_size), desc="Caching Mole-BERT", unit="batch"):
            graphs, valid = [], []
            for smiles in pending[start : start + args.ligand_batch_size]:
                try:
                    graphs.append(featurize(smiles))
                    valid.append(smiles)
                except Exception as exc:
                    failures[smiles] = f"{type(exc).__name__}: {exc}"
            if graphs:
                for smiles, vector in zip(valid, molebert_vectors(model, graphs, device)):
                    save_npy_atomic(ligand_cache_path(args.embeddings_dir, smiles), vector)
                    written += 1
        save_json(failures_path, {"failures": [{"smiles": s, "error": e} for s, e in sorted(failures.items())]})
    return {"total": len(smiles_values), "written": written, "failed": sum(s in failures for s in smiles_values)}


def main():
    parser = argparse.ArgumentParser(description="Cache ProtT5 and Mole-BERT vectors for MPEK EMULaToR splits.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--prottrans_path", type=str, default=PROTEIN_MODEL_ID)
    parser.add_argument("--no_catapro", action="store_true", help=f"Compute every ProtT5 vector. By default identical vectors are copied from {CATAPRO_PROTEIN_CACHE_DIR} (read only).")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ligand_batch_size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit_rows", type=int, default=0, help="Smoke-test option: read only this many rows per split file.")
    args = parser.parse_args()

    args.base_dir = Path(args.base_dir)
    args.embeddings_dir = Path(args.output_dir) / "embeddings"
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    jobs = discover_split_jobs(args.base_dir, tasks=args.tasks, split_groups=args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    sequences, smiles_values = collect_unique_values(jobs, args.sequence_col, args.smiles_col, args.target_col, args.limit_rows)
    print(f"Discovered {len(jobs)} split jobs, {len(sequences)} unique proteins, {len(smiles_values)} unique substrates")

    protein_stats = embed_proteins(args, sequences)
    ligand_stats = embed_ligands(args, smiles_values)
    save_json(
        args.embeddings_dir / "manifest.json",
        {
            "cache_version": CACHE_VERSION,
            "protein_max_length": PROTEIN_MAX_LENGTH,
            "prottrans_path": args.prottrans_path,
            "protein": protein_stats,
            "ligand": ligand_stats,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"Proteins: {protein_stats}\nLigands: {ligand_stats}")
    if ligand_stats["failed"]:
        print(f"Training drops the rows of {ligand_stats['failed']} ligands. See {args.embeddings_dir / FAILURES_NAME}.")


if __name__ == "__main__":
    main()
