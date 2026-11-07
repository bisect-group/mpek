"""The assay fast path (CUDA graphs and foreach optimizer) against the plain loop and against emulator_bench."""

from types import SimpleNamespace

import pytest
import torch

from assay_emulator_bench import train_single_split as assay
from assay_emulator_bench.modeling import CachedMPEKRegressor as AssayModel
from emulator_bench import train_single_split as emulator
from emulator_bench.fast_train import ForeachChildTuningAdamW, GraphedPredictor, GraphedTrainer
from emulator_bench.modeling import CachedMPEKRegressor as EmulatorModel
from MTLKcatKM.utils import ChildTuningAdamW

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEAD = dict(
    expert_dim=768,
    expert_layers=1,
    num_experts=4,
    ple_layers=1,
    dropout=0.3,
    tower_layers=2,
    tower_hidden=128,
    tower_dropout=0.0,
)
STEPS = 16_100  # more than the 15,901 eager steps, so epoch 1 crosses the switch and epoch 2 is graph replays only
SEED = 666


class Features(SimpleNamespace):
    def __len__(self):
        return len(self.protein)


def random_rows(n, device):
    g = torch.Generator().manual_seed(5)
    rows = (
        torch.randn(n, 1024, generator=g),
        torch.randn(n, 300, generator=g),
        1 + 10 * torch.rand(n, generator=g),
        100 * torch.rand(n, generator=g),
        torch.randn(n, 1, generator=g),
    )
    return [t.to(device) for t in rows]


def result(model, optimizer, train_pred, predicted):
    state = [
        (s["step"], s["exp_avg"].clone(), s["exp_avg_sq"].clone())
        for s in optimizer.state.values()
        if s
    ]
    return train_pred.clone(), predicted, [p.detach().clone() for p in model.parameters()], state


def run_assay(use_conditions, use_graph, rows):
    device = rows[0].device
    protein, ligand, pH, temperature, target = rows
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = AssayModel(use_conditions=use_conditions, device=device, **HEAD).to(device)
    optimizer = (ForeachChildTuningAdamW if use_graph else ChildTuningAdamW)(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6
    )
    inputs = [protein, ligand, pH, temperature]
    if use_graph:
        trainer = GraphedTrainer(
            model, optimizer, protein, ligand, target, lr=1e-4, extra=inputs[2:]
        )
    for _ in range(2):
        order = emulator.epoch_order(len(target))
        pred = (
            trainer.run(order.to(device))
            if use_graph
            else assay.train_epoch(model, optimizer, inputs, target, order, 1)
        )
    n = 1000
    if use_graph:
        predicted = GraphedPredictor(
            model, protein[:n], ligand[:n], extra=[pH[:n], temperature[:n]]
        ).predict()
    else:
        val = Features(
            protein=protein[:n],
            ligand=ligand[:n],
            pH=pH[:n],
            temperature=temperature[:n],
            target_std=1.0,
            target_mean=0.0,
        )
        predicted = assay.predict(model, val, device, 1).to(device).unsqueeze(1)
    return result(model, optimizer, pred, predicted)


def assert_identical(a, b):
    assert torch.equal(a[0], b[0]), "train predictions differ"
    assert torch.equal(a[1], b[1]), "eval predictions differ"
    assert all(torch.equal(x, y) for x, y in zip(a[2], b[2])), "weights differ"
    assert len(a[3]) == len(b[3]) and all(
        x[0] == y[0] == 2 * STEPS and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2])
        for x, y in zip(a[3], b[3])
    ), "Adam state differs"


@pytest.mark.parametrize(
    "use_conditions", [True, False], ids=["with_conditions", "without_conditions"]
)
def test_assay_fast_path_equals_plain_loop(use_conditions):
    rows = random_rows(STEPS, torch.device("cuda:0"))
    assert_identical(run_assay(use_conditions, False, rows), run_assay(use_conditions, True, rows))


def test_assay_without_conditions_equals_emulator_fast_path():
    device = torch.device("cuda:0")
    rows = random_rows(STEPS, device)
    protein, ligand, _, _, target = rows
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = EmulatorModel(**HEAD).to(device)
    optimizer = ForeachChildTuningAdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6
    )
    trainer = GraphedTrainer(model, optimizer, protein, ligand, target, lr=1e-4)
    for _ in range(2):
        pred = trainer.run(emulator.epoch_order(STEPS).to(device))
    predicted = GraphedPredictor(model, protein[:1000], ligand[:1000]).predict()
    assert_identical(result(model, optimizer, pred, predicted), run_assay(False, True, rows))
