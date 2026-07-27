import importlib.util
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from MTLKcatKM.layers import MLP, PLE


@lru_cache(maxsize=1)
def load_native_auxiliary_encoder():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "MTLKcatKM"
        / "encoder"
        / "auxiliary_encoder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_mpek_native_auxiliary_encoder", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MPEK auxiliary encoder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AuxiliaryEncoder


class CachedMPEKRegressor(nn.Module):
    def __init__(
        self,
        protein_dim: int = 1024,
        ligand_dim: int = 300,
        expert_dim: int = 768,
        expert_layers: int = 1,
        num_experts: int = 4,
        ple_layers: int = 1,
        dropout: float = 0.2,
        tower_layers: int = 3,
        tower_hidden: int = 128,
        tower_dropout: float = 0.0,
        use_assay_conditions: bool = False,
        condition_device: torch.device | str | None = None,
    ):
        super().__init__()
        self.use_assay_conditions = bool(use_assay_conditions)
        self.condition_device = torch.device(condition_device or "cpu")
        self.protein_projection = nn.Linear(protein_dim, ligand_dim)
        if self.use_assay_conditions:
            auxiliary_encoder = load_native_auxiliary_encoder()
            self.auxiliary_encoder = auxiliary_encoder(
                organism_dict_size=0,
                embed_size=ligand_dim,
                device=self.condition_device,
                use_ph=True,
                use_temperature=True,
                use_organism=False,
            )
        else:
            self.auxiliary_encoder = None
        self.ple_input_dim = ligand_dim * (3 if self.use_assay_conditions else 2)
        self.multi_block = PLE(
            experts_in=self.ple_input_dim,
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

    def forward(self, protein, ligand, pH=None, temperature=None):
        protein = self.protein_projection(protein)
        inputs = [ligand, protein]
        if self.use_assay_conditions:
            if pH is None or temperature is None:
                raise ValueError(
                    "pH and temperature are required when assay conditions are enabled"
                )
            condition = {"pH": pH, "temperature": temperature}
            auxiliary = self.auxiliary_encoder(None, condition)
            inputs.append(auxiliary)
        x = torch.cat(inputs, dim=1)
        tower_input = self.multi_block(x)[0]
        return self.tower(tower_input)
