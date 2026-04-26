import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, GATv2Conv

from torch_geometric.nn import GATConv



class GATLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3,
                 heads=4, dropout=0.3, edge_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        if in_channels != hidden_channels:
            self.input_proj = nn.Linear(in_channels, hidden_channels)
        else:
            self.input_proj = nn.Identity()

        for i in range(num_layers):
            in_dim = hidden_channels if i > 0 else hidden_channels
            out_dim = hidden_channels
            conv = GATConv(in_dim, out_dim, heads=heads, dropout=dropout,
                           concat=False, edge_dim=edge_dim)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(out_dim))
            if i > 0 and hidden_channels == out_dim:
                self.skips.append(nn.Identity())
            else:
                self.skips.append(None)

        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_proj(x)

        for i, conv in enumerate(self.convs):
            x_in = x
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            if self.skips[i] is not None and x_in.shape == x.shape:
                x = x + x_in
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin_out(x)
        return x

class GATv2Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(
            in_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.conv2 = GATv2Conv(
            hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
import random

class GraphSAGE_Manual(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_neighbors=5):
        super().__init__()
        self.conv1 = nn.Linear(in_channels * 2, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.num_neighbors = num_neighbors
        self.adj_lists = None

    def set_adj_lists(self, adj_lists):
        self.adj_lists = adj_lists

    def forward(self, x, edge_index):
        """
        x: [N, d] feature matrix
        edge_index: [2, E] edge indices (used to build adjacency lists if not already built)
        Returns logits of shape [N]
        """

        if self.adj_lists is None:
            num_nodes = x.size(0)
            adj = [[] for _ in range(num_nodes)]

            for u, v in edge_index.t().cpu().numpy():
                adj[u].append(v)
                adj[v].append(u)

            for i in range(num_nodes):
                adj[i] = list(set(adj[i]) - {i})

            self.adj_lists = adj

        batch_size = 256
        all_logits = []

        for start in range(0, x.size(0), batch_size):
            batch_nodes = list(range(start, min(start + batch_size, x.size(0))))

            sampled_neighbors = []
            for node in batch_nodes:
                neigh = self.adj_lists[node]

                if len(neigh) > self.num_neighbors:
                    neigh = random.sample(neigh, self.num_neighbors)

                sampled_neighbors.append(neigh)

            all_nodes = set(batch_nodes)
            for neighs in sampled_neighbors:
                all_nodes.update(neighs)

            all_nodes = list(all_nodes)
            node_to_idx = {n: i for i, n in enumerate(all_nodes)}

            x_sub = x[all_nodes]

            batch_embs = []

            for i, node in enumerate(batch_nodes):
                self_feat = x_sub[node_to_idx[node]]

                neigh_feat = [
                    x_sub[node_to_idx[n]]
                    for n in sampled_neighbors[i]
                ]

                if neigh_feat:
                    neigh_mean = torch.stack(neigh_feat).mean(dim=0)
                else:
                    neigh_mean = torch.zeros_like(self_feat)

                combined = torch.cat([self_feat, neigh_mean], dim=0)

                h = torch.relu(self.conv1(combined))
                batch_embs.append(h)

            logits = self.fc(torch.stack(batch_embs)).squeeze(-1)
            all_logits.append(logits)

        return torch.cat(all_logits)
