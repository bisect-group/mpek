"""Equivalence of the emulator_bench wrapper with the original MPEK code.

Run from baselines/mpek (see emulator_bench/verify_equivalence.sh):
    CUDA_VISIBLE_DEVICES= python -m pytest tests -m 'not gpu'   # CPU, tiny random T5, committed fixture
    python -m pytest tests -m gpu -s                              # CUDA graphs, real caches and splits on one GPU

The shipped original cannot run as it is: MTLKcatKM/layers has no MMoE (model.py imports it), `GNN_graphpred`
needs the author's modified copy, and `train.main` passes `normalizer=` to `evaluate`. The tests call the original
functions and classes unchanged and add only these shims.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(ROOT))

from emulator_bench import cache_embeddings, common  # noqa: E402  (adds MTLKcatKM and the Mole_BERT alias to sys.modules)
from emulator_bench import fast_train, train_single_split as train  # noqa: E402
from emulator_bench.dataset import CachedMPEKDataset  # noqa: E402
from emulator_bench.modeling import CachedMPEKRegressor  # noqa: E402

import MTLKcatKM.layers as og_layers  # noqa: E402

og_layers.MMoE = type("MMoE", (), {})  # model.py imports a layer the release does not ship
import MTLKcatKM.train as og_train  # noqa: E402
from MTLKcatKM.datasets import MTLKDataset  # noqa: E402
from MTLKcatKM.encoder import ligand_encoder  # noqa: E402
from MTLKcatKM.encoder.protein_encoder import ProteinEncoder  # noqa: E402
from MTLKcatKM.model import MTLModel  # noqa: E402
from MTLKcatKM.utils import ChildTuningAdamW, TrainNormalizer  # noqa: E402

HEAD = dict(expert_dim=768, expert_layers=1, num_experts=4, ple_layers=1, dropout=0.3, tower_layers=2, tower_hidden=128, tower_dropout=0.0)


class Rows(torch.utils.data.Dataset):
    """Cached features in the batch layout of MTLKDataset, so the original loop runs on them."""

    def __init__(self, protein, ligand, target):
        self.protein, self.ligand, self.target = protein, ligand, target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        return {"input_ids": self.protein[i], "attention_mask": torch.ones(1), "mol_graph": self.ligand[i], "labels": {"y": self.target[i]}}


class Features:
    """What `train.predict` reads from a dataset."""

    target_std, target_mean = 1.0, 0.0

    def __init__(self, protein, ligand):
        self.protein, self.ligand = protein, ligand

    def __len__(self):
        return len(self.protein)


class PassThrough(torch.nn.Module):
    """Stands for a frozen encoder: returns the cached vector it is given."""

    embed_dim = None

    def __init__(self, dim):
        super().__init__()
        self.embed_dim = dim

    def forward(self, features, *unused):
        return features


def original_model():
    return MTLModel(
        ligand_enc=PassThrough(300), protein_enc=PassThrough(1024), auxiliary_enc=None, expert_out=768, expert_hidden=768,
        expert_layers=1, num_experts=4, num_tasks=1, tower_hid_layer=2, tower_hid_unit=128, tower_dropout=[0.0], dropout=0.3,
        use_ph=False, use_temperature=False, use_organism=False, num_ple_layers=1, weights=[1],
    )


def random_rows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1024, generator=g), torch.randn(n, 300, generator=g), torch.randn(n, 1, generator=g)


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(FIXTURES / "tiny_table.csv", keep_default_na=False)


@pytest.fixture(scope="module")
def tiny_t5():
    """Random tiny T5 encoder and a committed tiny sentencepiece tokenizer, shared by original and wrapper."""
    from transformers import T5Config, T5EncoderModel, T5Tokenizer

    torch.manual_seed(0)
    model = T5EncoderModel(T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4)).eval()
    tokenizer = T5Tokenizer.from_pretrained(FIXTURES / "tiny_t5_tokenizer", extra_ids=0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(T5EncoderModel, "from_pretrained", lambda *a, **kw: model)
        mp.setattr(T5Tokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
        yield tokenizer


def original_dataset(table, tokenizer, max_length):
    """MTLKDataset.__getitem__ on the table rows (its constructor reads files with other column rules)."""
    ds = MTLKDataset.__new__(MTLKDataset)
    ds.max_length, ds.tokenizer = max_length, tokenizer
    ds.sequence_data, ds.smiles_data = table["sequence"].to_numpy(), table["smiles"].to_numpy()
    ds.labels, ds.task_names, ds.task = np.zeros((len(table), 1), dtype=np.float32), ["y"], "regression"
    ds.conversion, ds.use_ph, ds.use_temperature, ds.use_organism = 1, False, False, False
    return ds


def test_fixture_covers_cases(table):
    assert len(table) >= 60
    assert table["smiles"].str.contains(".", regex=False).sum() >= 3
    assert table["sequence"].str.contains("[UZOB]").sum() >= 6


def test_default_splits_cover_the_seven_folders():
    assert len(common.DEFAULT_SPLIT_GROUPS) == 7
    assert common.DEFAULT_SEEDS == [666, 777, 888]


def test_protein_inputs_equal_original(tiny_t5, table):
    long = "MKTAYIAKQR" * 130
    table = pd.concat([table, pd.DataFrame({"sequence": [long, long[::-1] + "UZOB", "MKTAYUZOBQ"], "smiles": ["CCO"] * 3})], ignore_index=True)
    for max_length in (1024, 64):  # 64 also truncates the fixture sequences
        original = original_dataset(table, tiny_t5, max_length)
        for i, sequence in enumerate(table["sequence"]):
            expected = original[i]
            ids = tiny_t5(common.protein_text(sequence), truncation=True, padding="max_length", max_length=max_length)
            assert ids["input_ids"] == expected["input_ids"].tolist() and ids["attention_mask"] == expected["attention_mask"].tolist()
    assert "X" in common.protein_text("AUZOBA") and common.protein_text("AUZOBA").count(" ") == 5


def test_protein_vectors_equal_original(tiny_t5, table, tmp_path):
    """Cache build against the original ProteinEncoder on the original dataset items, bit for bit."""
    long = "MKTAYIAKQR" * 130
    table = pd.concat([table.head(20), pd.DataFrame({"sequence": [long, "MKTAYUZOBQ"], "smiles": ["CCO"] * 2})], ignore_index=True)
    original = original_dataset(table, tiny_t5, common.PROTEIN_MAX_LENGTH)
    encoder = ProteinEncoder("unused", frozen_params=True)
    args = types.SimpleNamespace(embeddings_dir=tmp_path, overwrite=False, no_catapro=True, device="cpu", prottrans_path="unused")
    stats = cache_embeddings.embed_proteins(args, list(table["sequence"]))
    assert stats["computed"] == table["sequence"].nunique()
    for i, sequence in enumerate(table["sequence"]):
        item = original[i]
        expected = encoder(item["input_ids"][None], item["attention_mask"][None])[0].numpy()
        np.testing.assert_array_equal(np.load(common.protein_cache_path(tmp_path, common.protein_text(sequence))), expected)


def test_catapro_vectors_are_copied_only_for_identical_text(tiny_t5, tmp_path, monkeypatch):
    short, long = "MKTAYIAKQR" * 100, "MKTAYIAKQR" * 101  # 1000 and 1010 residues
    catapro = tmp_path / "catapro"
    catapro.mkdir()
    monkeypatch.setattr(common, "CATAPRO_PROTEIN_CACHE_DIR", catapro)
    for sequence in (short, long):
        np.save(common.catapro_protein_path(common.protein_text(sequence)), np.full(1024, 7, dtype=np.float32))
    args = types.SimpleNamespace(embeddings_dir=tmp_path / "mpek", overwrite=False, no_catapro=False, device="cpu", prottrans_path="unused")
    stats = cache_embeddings.embed_proteins(args, [short, long, "MKTAYUZOBQ"])
    assert (stats["copied_from_catapro"], stats["computed"]) == (1, 2)
    assert (np.load(common.protein_cache_path(args.embeddings_dir, common.protein_text(short))) == 7).all()
    assert not (np.load(common.protein_cache_path(args.embeddings_dir, common.protein_text(long))) == 7).all()


@pytest.fixture(scope="module")
def original_ligand_encoder():
    """The original LigandEncoderMoleBert on a reconstruction of the author's GNN_graphpred copy: the upstream class
    with the mean-pooled node vectors as output. The copy itself is not available."""
    from Mole_BERT.model import GNN_graphpred

    class Fork(GNN_graphpred):
        def __init__(self, num_layer, emb_dim, drop_ratio=0):
            super().__init__(num_layer, emb_dim, num_tasks=1, drop_ratio=drop_ratio)

        def from_pretrained(self, model_file, device=None):
            self.gnn.load_state_dict(torch.load(model_file, map_location=device))

        def forward(self, data):
            return self.pool(self.gnn(data.x, data.edge_index, data.edge_attr), data.batch)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ligand_encoder, "GNN_graphpred", Fork)
        yield ligand_encoder.LigandEncoderMoleBert(init_model=str(common.MOLE_BERT_ROOT / "model_gin" / "Mole-BERT.pth"), frozen_params=True)


def test_molebert_vectors_equal_original_encoder(original_ligand_encoder, table):
    from torch_geometric.loader import DataLoader

    smiles = sorted(set(table["smiles"]))
    graphs = [cache_embeddings.featurize(s) for s in smiles]
    model = cache_embeddings.load_molebert(torch.device("cpu"))
    got = cache_embeddings.molebert_vectors(model, graphs, torch.device("cpu"))
    expected = np.concatenate([original_ligand_encoder(batch).numpy() for batch in DataLoader(graphs, batch_size=1)])
    assert got.shape == (len(smiles), 300)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
    print(f"Mole-BERT, {len(smiles)} molecules: max abs diff {np.abs(got - expected).max():.2e}")


def test_unembeddable_ligands_are_dropped_and_counted(tiny_t5, table, tmp_path):
    frame = table.head(12).copy()
    frame.loc[3, "smiles"] = "not a smiles"
    frame["log10_value"] = frame["log10_value"].astype(float)
    path = tmp_path / "split.parquet"
    frame.to_parquet(path)
    args = types.SimpleNamespace(embeddings_dir=tmp_path, overwrite=False, no_catapro=True, device="cpu", prottrans_path="unused", ligand_batch_size=5)
    cache_embeddings.embed_proteins(args, list(frame["sequence"]))
    stats = cache_embeddings.embed_ligands(args, sorted(set(frame["smiles"])))
    assert stats["failed"] == 1
    ds = CachedMPEKDataset(str(path), str(tmp_path))
    assert (len(ds), ds.rows_dropped) == (11, 1) and ds.ligand.shape == (11, 300)
    again = cache_embeddings.embed_ligands(args, sorted(set(frame["smiles"])))
    assert again["written"] == 0 and again["failed"] == 1


def test_targets_equal_original_normalizer(tiny_t5, table, tmp_path):
    frame = table.copy()
    frame["log10_value"] = frame["log10_value"].astype(float)
    frame.to_parquet(tmp_path / "split.parquet")
    frame[["log10_value"]].to_csv(tmp_path / "train.csv", index=False)
    args = types.SimpleNamespace(embeddings_dir=tmp_path, overwrite=False, no_catapro=True, device="cpu", prottrans_path="unused", ligand_batch_size=64)
    cache_embeddings.embed_proteins(args, list(frame["sequence"]))
    cache_embeddings.embed_ligands(args, sorted(set(frame["smiles"])))
    ds = CachedMPEKDataset(str(tmp_path / "split.parquet"), str(tmp_path))
    normalizer = TrainNormalizer(str(tmp_path / "train.csv"), label_index=[0])
    expected = normalizer.norm(frame[["log10_value"]].to_numpy().astype(np.float32)).astype(np.float32)
    np.testing.assert_array_equal(ds.target.numpy(), expected)
    assert (ds.target_mean, ds.target_std) == (float(normalizer.mean[0, 0]), float(normalizer.std[0, 0]))


def test_model_init_equals_original():
    torch.manual_seed(666)
    original = original_model()
    torch.manual_seed(666)
    wrapper = CachedMPEKRegressor(**HEAD)
    expected, got = list(original.parameters()), list(wrapper.parameters())
    assert len(expected) == len(got) and sum(p.numel() for p in got) == sum(p.numel() for p in expected)
    for a, b in zip(expected, got):
        assert torch.equal(a, b)


def run_original(protein, ligand, target, epochs, seed, val):
    """The original `train` and `evaluate` with the DataLoader, ChildTuningAdamW and scheduler setup of `main`."""
    from torch.optim.lr_scheduler import LambdaLR
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    model = original_model()
    losses = []
    compute_loss = model.compute_loss

    def record_training_loss(*a):
        out = compute_loss(*a)
        if model.training:
            losses.append(out[0].detach().clone())
        return out

    model.compute_loss = record_training_loss
    optimizer = ChildTuningAdamW([{"params": list(model.parameters()), "lr": 1e-4}], betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = LambdaLR(optimizer, lambda x: 1.0)
    dataset = Rows(protein, ligand, target)
    train_loader = DataLoader(dataset, batch_sampler=torch.utils.data.BatchSampler(RandomSampler(dataset), 1, drop_last=True), shuffle=False)
    val_loader = DataLoader(Rows(*val), batch_size=1, shuffle=False)
    args = types.SimpleNamespace(use_organism=False, use_ph=False, use_temperature=False, distributed=False, disable_tqdm=True, log_interval=1000)
    val_preds = []
    for _ in range(epochs):
        og_train.train(model, train_loader, optimizer, torch.device("cpu"), args, scheduler)
        _, history = og_train.evaluate(model, val_loader, torch.device("cpu"), args)
        og_train.evaluate(model, DataLoader(Rows(*val), batch_size=1, shuffle=False), torch.device("cpu"), args)  # the test pass
        val_preds.append(torch.tensor(history["y_pred"]))
    return model, torch.stack(losses), val_preds


def test_training_steps_equal_original():
    """Per-step losses, weights after 3 epochs and validation predictions against the original train and evaluate."""
    rows, val, epochs, seed = random_rows(24, 1), random_rows(7, 2), 3, 666
    original, expected_losses, expected_val = run_original(*rows, epochs, seed, val)

    torch.manual_seed(seed)
    model = CachedMPEKRegressor(**HEAD)
    optimizer = ChildTuningAdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6)
    losses, device = [], torch.device("cpu")

    val_ds = Features(val[0], val[1])
    got_val = []
    for _ in range(epochs):
        order = train.epoch_order(len(rows[2]))
        protein, ligand, target = rows
        model.train()
        for idx in order.tolist():
            pred = model(protein[idx : idx + 1], ligand[idx : idx + 1]).float()
            loss = (pred - target[idx : idx + 1]).pow(2).sum() / (1 + 1e-5)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().clone())
        train.draw_loader_seed()
        got_val.append(train.predict(model, val_ds, device, 1))
        train.draw_loader_seed()
    assert torch.equal(torch.stack(losses), expected_losses)
    for a, b in zip(original.parameters(), model.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(expected_val, got_val):
        assert torch.equal(a.float(), b)
    print(f"{len(losses)} steps: losses, weights and validation predictions identical")


def test_train_epoch_function_equals_the_step_loop():
    """`train.train_epoch` (used by the run script) against the direct loop above, same RNG draws."""
    rows = random_rows(16, 3)
    results = []
    for use_function in (False, True):
        torch.manual_seed(5)
        model = CachedMPEKRegressor(**HEAD)
        optimizer = ChildTuningAdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6)
        order = train.epoch_order(16)
        if use_function:
            pred = train.train_epoch(model, optimizer, rows, order, 1)
        else:
            model.train()
            pred = torch.zeros_like(rows[2])
            for idx in order.tolist():
                out = model(rows[0][idx : idx + 1], rows[1][idx : idx + 1]).float()
                loss = (out - rows[2][idx : idx + 1]).pow(2).sum() / (1 + 1e-5)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                pred[idx] = out.detach()
        results.append((pred, [p.detach().clone() for p in model.parameters()]))
    assert torch.equal(results[0][0], results[1][0])
    assert all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_foreach_optimizer_equals_original_optimizer(device):
    rows = [t.to(device) for t in random_rows(40, 4)]
    outs = []
    for cls in (ChildTuningAdamW, fast_train.ForeachChildTuningAdamW):
        torch.manual_seed(7)
        model = CachedMPEKRegressor(**HEAD).to(device)
        optimizer = cls(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6)
        pred = train.train_epoch(model, optimizer, rows, train.epoch_order(40), 1)
        outs.append((pred, [p.detach().clone() for p in model.parameters()], [s["exp_avg_sq"].clone() for s in optimizer.state.values() if s]))
    assert torch.equal(outs[0][0], outs[1][0])
    for a, b in zip(outs[0][1] + outs[0][2], outs[1][1] + outs[1][2]):
        assert torch.equal(a, b)


def test_step_size_is_constant_in_fp32_after_the_stated_step():
    last = fast_train.constant_step_after(1e-4)
    t = np.arange(1, 2_000_000, dtype=np.float64)
    step_size = (1e-4 * np.sqrt(1 - 0.999**t) / (1 - 0.9**t)).astype(np.float32)
    assert last == 15900 and (step_size[last:] == np.float32(1e-4)).all() and step_size[last - 1] != np.float32(1e-4)


@pytest.mark.gpu
def test_cuda_graph_epoch_equals_plain_loop():
    """16,100 batch-1 steps: eager steps up to the constant step size, then graph replays. All train predictions,
    weights and Adam state equal the plain loop with the original optimizer, bit for bit. Predictions equal too."""
    n, device = 16_100, torch.device("cuda:0")
    rows = [t.to(device) for t in random_rows(n, 5)]
    runs = []
    for use_graph in (False, True):
        torch.manual_seed(666)
        torch.cuda.manual_seed_all(666)
        model = CachedMPEKRegressor(**HEAD).to(device)
        if use_graph:
            optimizer = fast_train.ForeachChildTuningAdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6)
            trainer = fast_train.GraphedTrainer(model, optimizer, *rows, lr=1e-4)
        else:
            optimizer = ChildTuningAdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6)
        for _ in range(2):  # the second epoch is graph replays only
            order = train.epoch_order(n)
            pred = trainer.run(order.to(device)).clone() if use_graph else train.train_epoch(model, optimizer, rows, order, 1)
        if use_graph:
            predicted = fast_train.GraphedPredictor(model, rows[0][:1000], rows[1][:1000]).predict()
        else:
            val = Features(rows[0][:1000], rows[1][:1000])
            predicted = train.predict(model, val, device, 1).to(device).unsqueeze(1)
        steps = {id(p): s for p, s in optimizer.state.items() if s}
        runs.append((pred, predicted, [p.detach().clone() for p in model.parameters()], [(s["step"], s["exp_avg"].clone(), s["exp_avg_sq"].clone()) for s in steps.values()]))
    plain, graph = runs
    assert torch.equal(plain[0], graph[0]), "train predictions differ"
    assert torch.equal(plain[1], graph[1]), "eval predictions differ"
    assert all(torch.equal(a, b) for a, b in zip(plain[2], graph[2])), "weights differ"
    assert all(a[0] == b[0] == 2 * n and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) for a, b in zip(plain[3], graph[3]))
    print(f"\nCUDA graphs vs plain loop over {2 * n} steps: train predictions, eval predictions, weights and Adam state identical")


@pytest.mark.gpu
def test_real_splits_and_caches():
    """The 42 runs exist, every unique text and SMILES is cached, and no ligand fails in the original loaders."""
    jobs = common.discover_split_jobs(common.DEFAULT_BASE_DIR)
    assert len(jobs) == 14 and {j["task"] for j in jobs} == {"kcat", "km"}
    sequences, smiles = cache_embeddings.collect_unique_values(jobs, "sequence", "smiles", "log10_value")
    emb = common.DEFAULT_EMBEDDINGS_DIR
    missing_proteins = [s for s in sequences if not common.protein_cache_path(emb, common.protein_text(s)).exists()]
    failed = {item["smiles"] for item in json.loads((emb / cache_embeddings.FAILURES_NAME).read_text())["failures"]} if (emb / cache_embeddings.FAILURES_NAME).exists() else set()
    missing_ligands = [s for s in smiles if s not in failed and not common.ligand_cache_path(emb, s).exists()]
    print(f"\n{len(jobs)} split jobs, {len(sequences)} sequences ({len(missing_proteins)} uncached), {len(smiles)} SMILES ({len(failed)} failed, {len(missing_ligands)} uncached)")
    assert not missing_proteins and not missing_ligands


@pytest.mark.gpu
def test_real_prott5_cache_equals_original_encoder():
    """Original ProteinEncoder (padded to 1024, batch 1, fp32) against the MPEK cache on a length-stratified sample
    that includes sequences over 1000 and over 1023 residues, and token ids of every reused sequence."""
    from transformers import T5Tokenizer

    jobs = common.discover_split_jobs(common.DEFAULT_BASE_DIR)
    sequences, _ = cache_embeddings.collect_unique_values(jobs, "sequence", "smiles", "log10_value")
    sequences = sorted(sequences, key=len)
    picks = {sequences[i] for i in np.linspace(0, len(sequences) - 1, 20).astype(int)}
    picks.update([s for s in sequences if 1000 < len(s) <= 1023][:3])
    picks.update([s for s in sequences if len(s) > 1023][::200][:4])
    device = torch.device("cuda:0")
    tokenizer, encoder = cache_embeddings.load_protein_encoder(common.PROTEIN_MODEL_ID, device)
    diffs = []
    for s in sorted(picks, key=len):
        text = common.protein_text(s)
        expected = encoder(*cache_embeddings.protein_inputs(tokenizer, text, device))[0].cpu().numpy()
        got = np.load(common.protein_cache_path(common.DEFAULT_EMBEDDINGS_DIR, text))
        diffs.append((len(s), np.abs(got - expected).max()))
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)
    assert any(n > 1023 for n, _ in diffs) and any(1000 < n <= 1023 for n, _ in diffs)
    legacy = T5Tokenizer.from_pretrained(common.PROTEIN_MODEL_ID, do_lower_case=False)
    reused = [s for s in sequences if len(s) <= common.CATAPRO_MAX_RESIDUES]
    mismatches = sum(
        tokenizer(common.protein_text(s), truncation=True, max_length=1024)["input_ids"] != legacy([common.protein_text(s)], add_special_tokens=True, padding="longest")["input_ids"][0]
        for s in reused
    )
    print(f"\nProtT5 cache vs original encoder: {len(diffs)} sequences, max abs diff {max(d for _, d in diffs):.2e}; CataPro token ids differ for {mismatches} of {len(reused)} reused sequences")
    assert mismatches == 0


@pytest.mark.gpu
def test_real_ligands_pass_the_original_loaders():
    from MTLKcatKM.datasets.dataset_test import MolTestDataset

    jobs = common.discover_split_jobs(common.DEFAULT_BASE_DIR)
    _, smiles = cache_embeddings.collect_unique_values(jobs, "sequence", "smiles", "log10_value")
    dataset = MolTestDataset.__new__(MolTestDataset)
    dataset.smiles_data, dataset.labels, dataset.task, dataset.conversion = smiles, np.zeros((len(smiles), 1)), "regression", 1
    for i in range(len(smiles)):
        dataset[i]  # the first graph MTLKDataset.__getitem__ builds
        cache_embeddings.featurize(smiles[i])  # the second one
