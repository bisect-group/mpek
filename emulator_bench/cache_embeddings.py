import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from emulator_bench.common import (
    CACHE_VERSION,
    DEFAULT_BASE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_GROUPS,
    DEFAULT_TASKS,
    MOLE_BERT_ROOT,
    PROTEIN_MAX_LENGTH,
    PROTEIN_MODEL_ID,
    discover_split_jobs,
    ensure_parent,
    ligand_cache_path,
    normalize_sequence,
    protein_cache_path,
    require_columns,
    save_json,
)
from emulator_bench.prepare_assets import validate_molebert
from src.utils.rich_progress import progress


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def save_npz_atomic(path: Path, payload: dict) -> None:
    ensure_parent(path)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(path)


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
            sequences.update(normalize_sequence(value) for value in frame[sequence_col].astype(str))
            smiles_values.update(str(value).strip() for value in frame[smiles_col].astype(str))
    return sorted(sequences), sorted(smiles_values)


def load_prott5(prottrans_path: str, device: torch.device):
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(prottrans_path, do_lower_case=False, legacy=False)
    model = T5EncoderModel.from_pretrained(prottrans_path).to(device)
    model.eval()
    return tokenizer, model


def embed_proteins(args, sequences):
    pending = [seq for seq in sequences if args.overwrite or not protein_cache_path(args.embeddings_dir, seq).exists()]
    if not pending:
        print("Protein cache is already complete.")
        return {"total": len(sequences), "written": 0}

    device = torch.device(args.device)
    print(f"Loading ProtT5 on {device}")
    tokenizer, model = load_prott5(args.prottrans_path, device)
    written = 0

    for start in progress(range(0, len(pending), args.protein_batch_size), desc="Caching ProtT5", unit="batch"):
        batch = pending[start : start + args.protein_batch_size]
        spaced = [" ".join(seq) for seq in batch]
        tokens = tokenizer(
            spaced,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=PROTEIN_MAX_LENGTH,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
        attention = tokens["attention_mask"]
        for index, sequence in enumerate(batch):
            seq_len = int(attention[index].sum().item())
            pooled = hidden[index, : max(seq_len - 1, 1)].mean(dim=0).detach().float().cpu().numpy()
            save_npz_atomic(
                protein_cache_path(args.embeddings_dir, sequence),
                {
                    "embedding": pooled,
                    "sequence": sequence,
                    "sequence_len": np.array([len(sequence)], dtype=np.int32),
                    "model": np.array(["ProtT5"], dtype=object),
                },
            )
            written += 1
    return {"total": len(sequences), "written": written}


def load_molebert(device: torch.device):
    validate_molebert()
    if str(MOLE_BERT_ROOT) not in sys.path:
        sys.path.insert(0, str(MOLE_BERT_ROOT))
    from model import GNN

    checkpoint = MOLE_BERT_ROOT / "model_gin" / "Mole-BERT.pth"
    model = GNN(5, 300, JK="last", drop_ratio=0.0, gnn_type="gin")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def safe_molebert_graph(mol: Chem.Mol) -> Data:
    """Build Mole-BERT input features while tolerating newer RDKit enum values."""
    atom_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in ATOM_LIST:
            raise ValueError(f"unsupported_atomic_number_{atomic_num}")
        chiral_tag = atom.GetChiralTag()
        chiral_index = CHIRALITY_LIST.index(chiral_tag) if chiral_tag in CHIRALITY_LIST else 0
        atom_features.append([ATOM_LIST.index(atomic_num), chiral_index])
    x = torch.tensor(np.array(atom_features), dtype=torch.long)

    edge_features = []
    edges = []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type not in BOND_LIST:
            raise ValueError(f"unsupported_bond_type_{bond_type}")
        bond_dir = bond.GetBondDir()
        if bond_dir not in BONDDIR_LIST:
            bond_dir = Chem.rdchem.BondDir.NONE
        edge_feature = [BOND_LIST.index(bond_type), BONDDIR_LIST.index(bond_dir)]
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(begin, end), (end, begin)])
        edge_features.extend([edge_feature, edge_feature])

    if edges:
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def validate_molebert_batch(batch: Batch, smiles_values) -> None:
    max_atom = int(batch.x[:, 0].max().item()) if batch.x.numel() else -1
    max_chiral = int(batch.x[:, 1].max().item()) if batch.x.numel() else -1
    max_bond = int(batch.edge_attr[:, 0].max().item()) if batch.edge_attr.numel() else -1
    max_dir = int(batch.edge_attr[:, 1].max().item()) if batch.edge_attr.numel() else -1
    if max_atom >= 120 or max_chiral >= 3 or max_bond >= 6 or max_dir >= 3:
        raise ValueError(
            "Mole-BERT feature index out of range for batch. "
            f"max_atom={max_atom}, max_chiral={max_chiral}, max_bond={max_bond}, max_dir={max_dir}, "
            f"first_smiles={smiles_values[:5]}"
        )


def embed_ligands(args, smiles_values):
    pending = [smiles for smiles in smiles_values if args.overwrite or not ligand_cache_path(args.embeddings_dir, smiles).exists()]
    if not pending:
        print("Ligand cache is already complete.")
        return {"total": len(smiles_values), "written": 0, "failed": 0}

    device = torch.device(args.device)
    model = load_molebert(device)
    written, failed = 0, 0
    failures = []

    for start in progress(range(0, len(pending), args.ligand_batch_size), desc="Caching Mole-BERT", unit="batch"):
        batch_smiles = pending[start : start + args.ligand_batch_size]
        graphs, valid_smiles = [], []
        for smiles in batch_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failures.append({"smiles": smiles, "error": "rdkit_parse_failed"})
                failed += 1
                continue
            try:
                graphs.append(safe_molebert_graph(mol))
                valid_smiles.append(smiles)
            except Exception as exc:
                failures.append({"smiles": smiles, "error": str(exc)})
                failed += 1
        if not graphs:
            continue
        batch = Batch.from_data_list(graphs).to(device)
        validate_molebert_batch(batch, valid_smiles)
        with torch.no_grad():
            node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
            graph_rep = global_mean_pool(node_rep, batch.batch).detach().float().cpu().numpy()
        for smiles, embedding in zip(valid_smiles, graph_rep):
            save_npz_atomic(
                ligand_cache_path(args.embeddings_dir, smiles),
                {
                    "embedding": embedding.astype(np.float32),
                    "smiles": np.array([smiles], dtype=object),
                    "model": np.array(["Mole-BERT"], dtype=object),
                },
            )
            written += 1
    if failures:
        save_json(args.embeddings_dir / "ligand_cache_failures.json", {"failures": failures})
    return {"total": len(smiles_values), "written": written, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Cache reusable ProtT5 and Mole-BERT embeddings for MPEK EMULaToR splits.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--prottrans_path", type=str, default=PROTEIN_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--protein_batch_size", type=int, default=1)
    parser.add_argument("--ligand_batch_size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit_rows", type=int, default=0, help="Debug/smoke mode: read only this many rows per split file.")
    args = parser.parse_args()

    args.base_dir = Path(args.base_dir)
    args.embeddings_dir = Path(args.output_dir) / "embeddings"
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    jobs = discover_split_jobs(args.base_dir, tasks=args.tasks, split_groups=args.split_groups)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    sequences, smiles_values = collect_unique_values(jobs, args.sequence_col, args.smiles_col, args.target_col, args.limit_rows)
    print(f"Discovered {len(jobs)} split jobs")
    print(f"Unique proteins: {len(sequences)}")
    print(f"Unique substrates: {len(smiles_values)}")

    protein_stats = embed_proteins(args, sequences)
    ligand_stats = embed_ligands(args, smiles_values)
    manifest = {
        "cache_version": CACHE_VERSION,
        "protein_max_length": PROTEIN_MAX_LENGTH,
        "base_dir": str(args.base_dir),
        "embeddings_dir": str(args.embeddings_dir),
        "tasks": list(args.tasks),
        "split_groups": list(args.split_groups),
        "sequence_col": args.sequence_col,
        "smiles_col": args.smiles_col,
        "target_col": args.target_col,
        "prottrans_path": args.prottrans_path,
        "molebert_root": str(MOLE_BERT_ROOT),
        "protein": protein_stats,
        "ligand": ligand_stats,
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.embeddings_dir / "manifest.json", manifest)
    print(f"Saved manifest to {args.embeddings_dir / 'manifest.json'}")
    if ligand_stats["failed"]:
        raise RuntimeError(f"{ligand_stats['failed']} ligands could not be embedded. See {args.embeddings_dir / 'ligand_cache_failures.json'}. Training on rows with these ligands fails with a missing-embedding error.")


if __name__ == "__main__":
    main()
