import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.nn import RGCNConv, global_add_pool
from torch_geometric.data import Data, Batch

_SRC_DIR     = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent
CHECKPOINTS_DIR = _PROJECT_DIR / "checkpoints"

class CircuitFilterGNN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=128, num_relations=22,
                 num_bases=8, num_layers=2):

        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        prev = in_channels
        for _ in range(num_layers):
            self.convs.append(RGCNConv(prev, hidden_channels, num_relations, num_bases=num_bases, aggr = 'add'))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            prev = hidden_channels

        self._graph_dim = in_channels + num_layers * hidden_channels
        concat_dim      = 2 * self._graph_dim

        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, data) -> torch.Tensor:

        x, edge_index, edge_type, batch = (
            data.x, data.edge_index, data.edge_type, data.batch
        )

        hop_pools = [global_add_pool(x, batch)]

        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, edge_type)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=0.3, training=self.training)
            hop_pools.append(global_add_pool(h, batch))

        return torch.cat(hop_pools, dim=1)

    def forward(self, candidate, target) -> torch.Tensor:

        h_cand = self.encode(candidate)
        h_tgt  = self.encode(target)
        h      = torch.cat([h_cand, h_tgt], dim=1)
        return self.mlp(h)

    @torch.no_grad()
    def predict(self, candidate, target) -> torch.Tensor:

        was_training = self.training
        self.eval()
        out = torch.sigmoid(self.forward(candidate, target))
        if was_training:
            self.train()
        return out

if __name__ == "__main__":
    model = CircuitFilterGNN()
    sep = "─" * 52
    print(f"\n{sep}\n  CircuitFilterGNN — target-conditioned (model.py)\n{sep}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {total:,}")
    print(f"  Layers (L)       : {model.num_layers}")
    print(f"  Per-graph dim    : {model._graph_dim}")
    print(f"  MLP input dim    : {2 * model._graph_dim}  ([h_K-hop ; h_target])")

    def fake(nnodes):
        d = Data(
            x=torch.randn(nnodes, 5),
            edge_index=torch.randint(0, nnodes, (2, nnodes * 2)),
            edge_type=torch.randint(0, 22, (nnodes * 2,)),
        )
        d.num_nodes = nnodes
        return d

    cand = Batch.from_data_list([fake(6), fake(8)])
    tgt  = Batch.from_data_list([fake(4), fake(4)])

    model.train()
    out = model(cand, tgt)
    assert out.shape == (2, 1), out.shape
    print(f"  Train pass OK    : logits {out.squeeze().tolist()}")

    prob = model.predict(cand, tgt)
    assert prob.shape == (2, 1) and (0 <= prob).all() and (prob <= 1).all()
    assert model.training, "predict() failed to restore training mode"
    print(f"  Eval  pass OK    : probs  {[round(p,4) for p in prob.squeeze().tolist()]}")
    print(f"{sep}\n")