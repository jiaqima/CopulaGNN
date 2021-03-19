from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import time
from six.moves import cPickle as pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import (generate_lsn, to_data, read_election, read_wiki, read_emnlp)
from models import (MLP, GCN, SAGE, GAT, APPNPNet, CorCopulaGCN, CorCopulaSAGE,
                    RegCopulaGCN, RegCopulaSAGE,)
from utils import Logger

import warnings
warnings.filterwarnings('error')

parser = argparse.ArgumentParser(description='Main.')
parser.add_argument("--verbose", type=int, default=2)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_trials", type=int, default=10)

# Dataset configuration
parser.add_argument("--path", default="./data")
parser.add_argument("--dataset", default="wiki-squirrel")
# Synthetic data configuration
parser.add_argument(
    "--lsn_mode", default="daxw",
    help=("Choices: `daxwi`, `xw', or `daxw`. \n"
          "  `daxwi`: only mean is graph-dependent; \n"
          "  `xw`: only cov is graph-dependent; \n"
          "  `daxw`: both mean and cov are graph-dependent."))
parser.add_argument("--num_features", type=int, default=10)
parser.add_argument("--num_nodes", type=int, default=300)
parser.add_argument("--num_edges", type=int, default=5000)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--tau", type=float, default=1.0)

# Model configuration
parser.add_argument("--model_type", default="mlp")
parser.add_argument("--hidden_size", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--clip_output", type=float, default=0.5)

# Training configuration
parser.add_argument("--opt", default="Adam")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10000)
parser.add_argument("--patience", type=int, default=50)

# Other configuration
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("--result_path", default=None)

args = parser.parse_args()

# Set random seed
if args.seed >= 0:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)

# Load data
data_seed = int(np.ceil(args.seed / float(args.num_trials)))
if args.dataset == "lsn":
    x, y, adj, datafile = generate_lsn(n=args.num_nodes,
                                       d=args.num_features,
                                       m=args.num_edges,
                                       gamma=args.gamma,
                                       tau=args.tau,
                                       seed=data_seed,
                                       lsn_mode=args.lsn_mode,
                                       root=args.path,
                                       save_file=False)
    data = to_data(x, y, adj=adj)
    data.is_count_data = False
    data.to(args.device)
elif args.dataset.startswith("election"):
    target = args.dataset.split("-")[1]
    data = read_election("data", target, seed=data_seed)
    data.is_count_data = False
    data.to(args.device)
elif args.dataset.startswith("wiki"):
    name = args.dataset.split("-")[1]
    data = read_wiki("data", name=name, seed=data_seed)
    data.is_count_data = True
    data.to(args.device)
elif args.dataset.startswith("emnlp", seed=data_seed):
    data = read_emnlp("data")
    data.is_count_data = True
    data.to(args.device)
else:
    raise NotImplementedError("Dataset {} is not supported.".format(
        args.dataset))

# Outcome type config
if not data.is_count_data:
    marginal_type = "Normal"

    # R-squared
    def metric(preds, labels):
        num = torch.mean((preds - labels)**2)
        denum = torch.mean((labels - torch.mean(labels))**2)
        return 1 - num / denum
else:
    marginal_type = "Poisson"

    # R-squared based on deviance residuals for count data
    # Suitable for heteroscedastic data
    # http://cameron.econ.ucdavis.edu/research/jbes96preprint.pdf
    def metric(preds, labels):
        labels = 1 + labels
        preds = 1 + preds
        ratio = torch.log(labels / preds)
        num = torch.mean(labels * ratio - (labels - preds))
        denum = torch.mean(labels * torch.log(labels / torch.mean(labels)))
        return 1 - num / denum

minimize_metric = -1

# Log file
time_stamp = time.time()
log_file = (
    "data__{}__model__{}__lr__{}__h__{}__seed__{}__stamp__{}").format(
    args.dataset, args.model_type, args.lr, args.hidden_size, args.seed,
    time_stamp)
if args.dataset == "lsn":
    log_file += "__datafile__{}".format(os.path.splitext(datafile)[0])
log_path = os.path.join(args.path, "logs")
lgr = Logger(args.verbose, log_path, log_file)
lgr.p(args)

# Model config
model_args = {
    "num_features": data.x.size(1),
    "hidden_size": args.hidden_size,
    "dropout": args.dropout,
    "activation": "relu"
}

if args.model_type in ["corcgcn", "regcgcn", "corcsage", "regcsage"]:
    model_args["marginal_type"] = marginal_type

if args.model_type == "mlp":
    model = MLP(**model_args)
elif args.model_type == "gcn":
    model = GCN(**model_args)
elif args.model_type == "sage":
    model = SAGE(**model_args)
elif args.model_type == "gat":
    model_args["num_heads"] = args.num_heads
    model_args["hidden_size"] = int(args.hidden_size / args.num_heads)
    model = GAT(**model_args)
elif args.model_type == "appnp":
    model = APPNPNet(**model_args)
elif args.model_type == "corcgcn":
    model = CorCopulaGCN(**model_args)
elif args.model_type == "corcsage":
    model = CorCopulaSAGE(**model_args)
elif args.model_type == "regcgcn":
    model = RegCopulaGCN(**model_args)
elif args.model_type == "regcsage":
    model = RegCopulaSAGE(**model_args)
else:
    raise NotImplementedError("Model {} is not supported.".format(
        args.model_type))
model.to(args.device)

# Optimizer
if args.opt == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    raise NotImplementedError("Optimizer {} is not supported.".format(
        args.opt))

# Training objective
if hasattr(model, "nll"):

    def train_loss_fn(model, data):
        return model.nll(data)

else:

    if marginal_type == "Normal":
        criterion = nn.MSELoss()
    elif marginal_type == "Poisson":

        def criterion(logits, labels):
            return torch.mean(torch.exp(logits) - labels * logits)

    else:
        raise NotImplementedError("Marginal type {} is not supported.".format(
            marginal_type))

    def train_loss_fn(model, data):
        return criterion(
            model(data)[data.train_mask], data.y[data.train_mask])


# Training and evaluation
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
            preds = model.predict(data, num_samples=1000)
        else:
            preds = model(data)
            if marginal_type == "Poisson":
                preds = torch.exp(preds)
        if args.clip_output != 0:  # clip output logits to avoid extreme outliers
            left = torch.min(data.y[data.train_mask]) / args.clip_output
            right = torch.max(data.y[data.train_mask]) * args.clip_output
            preds = torch.clamp(preds, left, right)
        train_metric = metric(
            preds[data.train_mask], data.y[data.train_mask]).item()
        valid_metric = metric(
            preds[data.valid_mask], data.y[data.valid_mask]).item()
        test_metric = metric(
            preds[data.test_mask], data.y[data.test_mask]).item()
    return train_metric, valid_metric, test_metric


patience = args.patience
best_metric = np.inf
stats_to_save = {"args": args, "traj": []}
for epoch in range(args.num_epochs):
    train()
    if (epoch + 1) % args.log_interval == 0:
        train_metric, valid_metric, test_metric = test()
        this_metric = valid_metric * minimize_metric
        patience -= 1
        if this_metric < best_metric:
            patience = args.patience
            best_metric = this_metric
            stats_to_save["valid_metric"] = valid_metric
            stats_to_save["test_metric"] = test_metric
            stats_to_save["epoch"] = epoch
        stats_to_save["traj"].append({
            "epoch": epoch,
            "valid_metric": valid_metric,
            "test_metric": test_metric
            })
        if patience == 0:
            break
        lgr.p("Epoch {}: train {:.4f}, valid {:.4f}, test {:.4f}".format(
            epoch, train_metric, valid_metric, test_metric))

lgr.p("-----\nBest epoch {}: valid {:.4f}, test {:.4f}".format(
      stats_to_save["epoch"], stats_to_save["valid_metric"],
      stats_to_save["test_metric"]))

# Write outputs
if args.verbose == 0:
    if args.result_path is None:
        result_path = os.path.join(args.path, "results")
    else:
        result_path = args.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_file = (
        "data__{}__valid__{}__test__{}__model__{}__lr__{}__h__{}__seed__{}"
        "__stamp__{}").format(
        args.dataset, stats_to_save["valid_metric"],
        stats_to_save["test_metric"], args.model_type, args.lr,
        args.hidden_size, args.seed, time_stamp)
    if args.dataset == "lsn":
        result_file += "__datafile__{}".format(os.path.splitext(datafile)[0])
    with open(os.path.join(result_path, result_file), "wb") as f:
        pickle.dump(stats_to_save, f)
