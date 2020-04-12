from __future__ import absolute_import, division, print_function

import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import load_data
from models import (GAT, GCN, GenGNN, MLP,
                    RegressionCGCN, SpectralCGCN)
from torch_geometric.utils import to_dense_adj

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Main.')
parser.add_argument("--verbose", type=int, default=2)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=10)

# Dataset configuration
parser.add_argument("--path", default="./data")
parser.add_argument("--dataset", default="lsn")
parser.add_argument("--mean_mode", default="daxw")
parser.add_argument("--num_features", type=int, default=10)
parser.add_argument("--num_nodes", type=int, default=300)
parser.add_argument("--num_edges", type=int, default=5000)
parser.add_argument("--gamma", type=float, default=0.05)
parser.add_argument("--tau", type=float, default=2)

# Model configuration.
parser.add_argument("--model_type", default="mlp")
parser.add_argument("--hidden_size", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.)
parser.add_argument("--m_gamma", type=float, default=-1)
parser.add_argument("--m_tau", type=float, default=-1)

# Training configuration.
parser.add_argument("--opt", default="Adam")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lamda", type=float, default=1e-2)

# Other configuration
parser.add_argument("--test_metric", default="mse")
parser.add_argument("--num_epochs", type=int, default=2000)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--result_path", default=None)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--model_path", default=None)
parser.add_argument("--save_log", action="store_true")
parser.add_argument("--log_path", default=None)

args = parser.parse_args()

if args.m_gamma < 0:
    args.m_gamma = args.gamma
if args.m_tau < 0:
    args.m_tau = args.m_tau

# Set random seed
if args.seed >= 0:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)

data = load_data(args.dataset)
adj = to_dense_adj(data.edge_index).cpu().numpy()
data.to(args.device)
criterion = nn.CrossEntropyLoss()

m_adj = adj

model_args = {
    "num_features": data.x.size(1),
    "num_classes": data.num_classes,
    "hidden_size": args.hidden_size,
    "dropout": args.dropout,
    "activation": "relu"
}

if "spectral" in args.model_type:
    model_args["adj"] = m_adj

if args.model_type in ["mlp"]:
    model = MLP(**model_args)
elif args.model_type in ["gcn"]:
    model = GCN(**model_args)
elif args.model_type == "gat":
    model = GAT(**model_args)
elif "_" in args.model_type:
    gen_type, post_type = args.model_type.split("_")

    gen_config = copy.deepcopy(model_args)
    gen_config["type"] = gen_type
    gen_config["neg_ratio"] = 1.0
    if gen_type == "lsm":
        gen_config["hidden_x"] = args.hidden_size

    post_config = copy.deepcopy(model_args)
    post_config["type"] = post_type
    # if post_type == "gat":
    #     post_config["num_heads"] = args.num_heads
    #     post_config["hidden_size"] = int(args.hidden / args.num_heads)
    model = GenGNN(gen_config, post_config)
elif args.model_type in ["spectralcgcn"]:
    model = SpectralCGCN(**model_args)
elif args.model_type in ["regressioncgcn"]:
    model = RegressionCGCN(**model_args)
else:
    raise NotImplementedError("Model {} is not supported.".format(
        args.model_type))
model.to(args.device)

if args.opt == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    raise NotImplementedError("Optimizer {} is not supported.".format(
        args.opt))

if hasattr(model, "gen"):

    if model.post_type in ["regressioncgcn"]:

        def train_loss_fn(model, data):
            post_y_pred = model(data)
            nll_generative = model.gen.nll_generative(data, post_y_pred)
            nll_discriminative = model.post.nll(data)
            return args.lamda * nll_generative + nll_discriminative

    else:

        def train_loss_fn(model, data):
            post_y_pred = model(data)
            nll_generative = model.gen.nll_generative(data, post_y_pred)
            nll_discriminative = criterion(post_y_pred[data.train_mask],
                                           data.y[data.train_mask])
            return args.lamda * nll_generative + nll_discriminative

elif hasattr(model, "nll"):

    def train_loss_fn(model, data):
        return model.nll(data)

else:

    def train_loss_fn(model, data):
        return criterion(model(data)[data.train_mask], data.y[data.train_mask])


def test_loss_fn(logits, data, mask):
    return criterion(logits[mask], data.y[mask]).item()


def train():
    model.train()
    optimizer.zero_grad()
    loss = train_loss_fn(model, data)
    loss.backward()
    optimizer.step()


def test():
    model.eval()
    with torch.no_grad():
        if hasattr(model, "predict"):
            logits = model.predict(data, num_samples=1000)
            # logits = model(data)
        elif hasattr(model, "post") and model.post_type in ["regressioncgcn"]:
            logits = model(data, num_samples=1000)
            # logits = model.post(data)
        else:
            logits = model(data)
        train_loss = test_loss_fn(logits, data, data.train_mask)
        valid_loss = test_loss_fn(logits, data, data.valid_mask)
        test_loss = test_loss_fn(logits, data, data.test_mask)
        accs = []
        for _, mask in data('train_mask', 'valid_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return train_loss, valid_loss, test_loss, accs


patience = args.patience
best_metric = np.inf
selected_metrics = []
model.train()
for epoch in range(args.num_epochs):
    train()
    if (epoch + 1) % args.log_interval == 0:
        train_loss, valid_loss, test_loss, accs = test()
        this_metric = valid_loss
        patience -= 1
        if this_metric < best_metric:
            patience = args.patience
            best_metric = this_metric
            selected_metrics = [valid_loss, test_loss]
        if patience == 0:
            break
        if args.verbose > 1:
            print("Epoch {}: train {:.4f}, valid {:.4f}, test {:.4f}".format(
                epoch, *accs))
