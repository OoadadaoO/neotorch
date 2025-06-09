import os
import random
import time
import argparse
import numpy as np
import torch
from torch.nn import (
    Module,
    ModuleList,
    LayerNorm,
    Linear,
    Dropout,
    ReLU,
    Identity,
    BatchNorm1d,
)
from torch_geometric.nn import SAGEConv


def transform_activation(activ) -> Module:
    if activ == "relu":
        return ReLU()
    elif activ == "sigmoid":
        return torch.sigmoid
    elif activ == "none":
        return Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activ}")


def transform_norm(norm, dim=None) -> Module:
    if norm == "layer_norm":
        return LayerNorm(dim, elementwise_affine=True)
    elif norm == "batch_norm":
        return BatchNorm1d(dim)
    elif norm == "none":
        return Identity()
    else:
        raise ValueError(f"Unsupported normalization: {norm}")


class GNNConv(Module):
    def __init__(self, in_dim, out_dim, aggr, activ, dropout, norm):
        super().__init__()
        self.norm = transform_norm(norm, in_dim)
        self.activate = transform_activation(activ)
        self.dropout = Dropout(dropout)
        self.conv = SAGEConv(in_dim, out_dim, aggr=aggr)

    def reset_parameters(self):
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activate(self.norm(x)))
        x = self.conv(x, edge_index)
        return x


class GNN(Module):
    """
    Liner_1(in) -> x
    loop {
        x -> u
        norm(u) -> u
        activate(u) -> u
        dropout(u) -> u
        conv(u) + x -> x
    }
    norm(x) -> x
    activate(x) -> x
    dropout(x) -> x
    Liner_2(x) -> out
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        aggr,
        activ,
        num_pre_linears,
        num_post_linears,
        dropout,
        norm,
        residual,
    ):
        super().__init__()
        self.residual = residual

        if num_pre_linears > 0:
            self.lin1 = ModuleList(
                [
                    Linear(in_dim if i == 0 else hidden_dim, hidden_dim)
                    for i in range(num_pre_linears)
                ]
            )
        else:
            self.lin1 = None

        self.convs = ModuleList()
        for i in range(num_layers):
            if i == 0 and num_pre_linears == 0:
                conv = GNNConv(in_dim, hidden_dim, aggr, activ, 0.0, "none")
            elif i == num_layers - 1 and num_post_linears == 0:
                conv = GNNConv(hidden_dim, out_dim, aggr, activ, dropout, norm)
            else:
                conv = GNNConv(hidden_dim, hidden_dim, aggr, activ, dropout, norm)
            self.convs.append(conv)

        self.norm = transform_norm(norm, hidden_dim)
        self.activate = transform_activation(activ)
        self.dropout = Dropout(dropout)

        if num_post_linears > 0:
            self.lin2 = ModuleList(
                [
                    Linear(
                        hidden_dim, out_dim if i == num_post_linears - 1 else hidden_dim
                    )
                    for i in range(num_post_linears)
                ]
            )
        else:
            self.lin2 = None

    def reset_parameters(self):
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()
        if isinstance(self.lin1, ModuleList):
            for lin in self.lin1:
                lin.reset_parameters()
        if isinstance(self.lin2, ModuleList):
            for lin in self.lin2:
                lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.lin1 is not None:
            for lin in self.lin1:
                x = lin(x)
        # Residual method: https://ar5iv.labs.arxiv.org/html/1603.05027
        for conv in self.convs:
            if self.residual:
                x = conv(x, edge_index) + x
            else:
                x = conv(x, edge_index)
        if self.lin2 is not None:
            x = self.dropout(self.activate(self.norm(x)))
            for lin in self.lin2:
                x = lin(x)
        return x


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--out_dim", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--activ", type=str, default="relu")
    parser.add_argument("--num_pre_linears", type=int, default=0)
    parser.add_argument("--num_post_linears", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--norm", type=str, default="none")
    parser.add_argument("--residual", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    model = GNN(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        aggr=args.aggr,
        activ=args.activ,
        num_pre_linears=args.num_pre_linears,
        num_post_linears=args.num_post_linears,
        dropout=args.dropout,
        norm=args.norm,
        residual=args.residual,
    )
    t = time.time()
    model.train()
    model = torch.jit.script(model)
    torch.jit.save(model, os.path.join(args.output_dir, f"{args.name}.pt"))
    print(f">>> 模型已成功編譯並保存為 '{args.name}.pt'")
    print(f">>> 編譯時間: {time.time() - t:.2f}秒")
