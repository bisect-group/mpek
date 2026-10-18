import torch
from torch import nn

from MTLKcatKM.layers import MLP, PLE


class CachedMPEKRegressor(nn.Module):
    """MPEK head on cached ProtT5 and Mole-BERT vectors, without pH, temperature or organism inputs."""

    def __init__(
        self,
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
        self.multi_block = PLE(
            experts_in=ligand_dim * 2,
            experts_out=expert_dim,
            experts_hidden=expert_dim,
            expert_hid_layer=expert_layers,
            dropout_rate=dropout,
            num_experts=num_experts,
            num_tasks=1,
            num_ple_layers=ple_layers,
        )
        self.tower = MLP(
            in_size=expert_dim,
            hidden_size=tower_hidden,
            out_size=1,
            layer_num=tower_layers,
            dropout_rate=tower_dropout,
        )

    def forward(self, protein, ligand):
        protein = self.protein_projection(protein)
        x = torch.cat([ligand, protein], dim=1)
        return self.tower(self.multi_block(x)[0])
