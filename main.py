from __future__ import absolute_import, division, print_function

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_lsn, to_data
from models import GATReg, GCNReg, MLPReg

parser = argparse.ArgumentParser(description='Main.')
parser.add_argument("--verbose", type=int, default=2)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=-1)

# Dataset configuration
parser.add_argument("--path", default="./data")
parser.add_argument("--dataset", default="lsn")
parser.add_argument("--num_features", type=int, default=10)
parser.add_argument("--num_nodes", type=int, default=300)
parser.add_argument("--num_edges", type=int, default=5000)

# Model configuration.
parser.add_argument("--model_type", default="mlp")
parser.add_argument("--hidden_size", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.)

# Training configuration.
parser.add_argument("--opt", default="Adam")
parser.add_argument("--lr", type=float, default=0.001)

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
        root=args.path,
        save_file=False)
    data = to_data(x, y, adj)
    data.to(args.device)
    criterion = nn.MSELoss()
else:
    raise NotImplementedError(
        "Dataset {} is not supported.".format(args.dataset))

num_features = data.x.size(1)

if args.model_type == "mlp":
    model = MLPReg(num_features=num_features, hidden_size=args.hidden_size)
elif args.model_type == "gcn":
    model = GCNReg(
        num_features=num_features,
        hidden_size=args.hidden_size,
        dropout=args.dropout)
elif args.model_type == "gat":
    model = GATReg(
        num_features=num_features,
        hidden_size=args.hidden_size,
        dropout=args.dropout)
else:
    raise NotImplementedError(
        "Model {} is not supported.".format(args.model_type))
model.to(args.device)

if args.opt == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError(
        "Optimizer {} is not supported.".format(args.opt))

patience = args.patience
best_metric = np.inf
model.train()
for epoch in range(args.num_epochs):
    logits = model(data)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % args.log_interval == 0:
        model.eval()
        with torch.no_grad():
            logits = model(data)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            valid_loss = criterion(logits[data.valid_mask],
                                   data.y[data.valid_mask])
            test_loss = criterion(logits[data.test_mask],
                                  data.y[data.test_mask])
        model.train()
        this_metric = valid_loss.item()
        patience -= 1
        if this_metric < best_metric:
            patience = args.patience
            best_metric = this_metric
        if patience == 0:
            break
        if args.verbose > 1:
            print("Epoch {}: train {:.2f}, valid {:.2f}, test {:.2f}".format(
                epoch, loss.item(), valid_loss.item(), test_loss.item()))
