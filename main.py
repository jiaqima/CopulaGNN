from __future__ import absolute_import, division, print_function

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_lsn, to_data
from models import CGCNReg, CMLPReg, GATReg, GCNReg, GenGNN, MLPReg

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

# Training configuration.
parser.add_argument("--opt", default="Adam")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lamda", type=float, default=1e-2)

# Other configuration
parser.add_argument("--num_epochs", type=int, default=2000)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--result_path", default=None)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--model_path", default=None)
parser.add_argument("--save_log", action="store_true")
parser.add_argument("--log_path", default=None)

args = parser.parse_args()

# Set random seed
if args.seed >= 0:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)

if args.dataset == "lsn":
    x, y, adj = generate_lsn(
        n=args.num_nodes,
        d=args.num_features,
        m=args.num_edges,
        gamma=args.gamma,
        tau=args.tau,
        seed=args.seed,
        mean_mode=args.mean_mode,
        root=args.path,
        save_file=True)
    data = to_data(x, y, adj)
    data.to(args.device)
    criterion = nn.MSELoss()
else:
    raise NotImplementedError(
        "Dataset {} is not supported.".format(args.dataset))

model_args = {
    "num_features": data.x.size(1),
    "hidden_size": args.hidden_size,
    "dropout": args.dropout,
    "activation": "relu"
}

if args.model_type == "mlp":
    model = MLPReg(**model_args)
elif args.model_type == "gcn":
    model = GCNReg(**model_args)
elif args.model_type == "gat":
    model = GATReg(**model_args)
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
elif args.model_type == "cmlp":
    model = CMLPReg(**model_args)
elif args.model_type == "cgcn":
    model = CGCNReg(**model_args)
else:
    raise NotImplementedError(
        "Model {} is not supported.".format(args.model_type))
model.to(args.device)

if args.opt == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError(
        "Optimizer {} is not supported.".format(args.opt))

if hasattr(model, "gen"):

    def train_loss_fn(model, data):
        post_y_pred = model(data)
        nll_generative = model.gen.nll_generative(data, post_y_pred)
        nll_discriminative = criterion(post_y_pred[data.train_mask],
                                       data.y[data.train_mask])
        return args.lamda * nll_generative + nll_discriminative
elif hasattr(model, "nll_copula"):
    L = np.diag(adj.sum(axis=0)) - adj
    cov = np.linalg.inv(L + args.gamma * np.eye(adj.shape[0]))
    cov = torch.tensor(cov, dtype=torch.float32).to(args.device)
    cov = cov[data.train_mask, :]
    cov = cov[:, data.train_mask]

    def train_loss_fn(model, data):
        pred = model(data)[data.train_mask]
        label = data.y[data.train_mask]
        nll_copula = model.nll_copula(pred, label, cov)
        nll_q = criterion(pred, label)
        return args.lamda * nll_copula + nll_q
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
        logits = model(data)
        train_loss = test_loss_fn(logits, data, data.train_mask)
        valid_loss = test_loss_fn(logits, data, data.valid_mask)
        test_loss = test_loss_fn(logits, data, data.test_mask)
    return train_loss, valid_loss, test_loss


patience = args.patience
best_metric = np.inf
selected_metrics = []
model.train()
for epoch in range(args.num_epochs):
    train()
    if (epoch + 1) % args.log_interval == 0:
        train_loss, valid_loss, test_loss = test()
        this_metric = valid_loss
        patience -= 1
        if this_metric < best_metric:
            patience = args.patience
            best_metric = this_metric
            selected_metrics = [valid_loss, test_loss]
        if patience == 0:
            break
        if args.verbose > 1:
            print("Epoch {}: train {:.2f}, valid {:.2f}, test {:.2f}".format(
                epoch, train_loss, valid_loss, test_loss))

if args.verbose == 0:
    with open(
            os.path.join(args.path, args.dataset, "results",
                         "valid__{}__test__{}__seed__{}__model__{}".format(
                             selected_metrics[0], selected_metrics[1],
                             args.seed, args.model_type)), "w") as f:
        pass
