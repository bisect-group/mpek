import importlib.util

import torch
from torch import nn

from assay_emulator_bench.common import MPEK_ROOT
from MTLKcatKM.layers import MLP, PLE


def load_auxiliary_encoder():
    """Load the original condition encoder by path; its package __init__ imports the full Mole-BERT stack."""
    spec = importlib.util.spec_from_file_location("mpek_auxiliary_encoder", MPEK_ROOT / "encoder" / "auxiliary_encoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AuxiliaryEncoder


class CachedMPEKRegressor(nn.Module):
    """MPEK head on cached ProtT5 and Mole-BERT vectors, with the original pH and temperature encoder when enabled."""

    def __init__(
        self,
        use_conditions: bool,
        device: torch.device,
        expert_dim: int,
        expert_layers: int,
        num_experts: int,
        ple_layers: int,
        dropout: float,
        tower_layers: int,
        tower_hidden: int,
        tower_dropout: float,
        protein_dim: int = 1024,
        ligand_dim: int = 300,
    ):
        super().__init__()
        self.protein_projection = nn.Linear(protein_dim, ligand_dim)
        self.condition_encoder = None
        if use_conditions:
            self.condition_encoder = load_auxiliary_encoder()(
                organism_dict_size=0, embed_size=ligand_dim, device=device, use_ph=True, use_temperature=True, use_organism=False
            )
        self.multi_block = PLE(
            experts_in=ligand_dim * (3 if use_conditions else 2),
            experts_out=expert_dim,
            experts_hidden=expert_dim,
            expert_hid_layer=expert_layers,
            dropout_rate=dropout,
            num_experts=num_experts,
            num_tasks=1,
            num_ple_layers=ple_layers,
        )
        self.tower = MLP(in_size=expert_dim, hidden_size=tower_hidden, out_size=1, layer_num=tower_layers, dropout_rate=tower_dropout)

    def forward(self, protein, ligand, pH, temperature):
        features = [ligand, self.protein_projection(protein)]
        if self.condition_encoder is not None:
            features.append(self.condition_encoder(None, {"pH": pH, "temperature": temperature}))
        return self.tower(self.multi_block(torch.cat(features, dim=1))[0])
