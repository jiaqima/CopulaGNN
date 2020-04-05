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
from models import (CGCNReg, CMLPReg, GATReg, GCNReg, GenGNN, MLPReg,
                    NewCGCNReg, NewCMLPReg, RegressionCGCNReg,
                    RegressionCMLPReg, SpectralCGCNReg, SpectralCMLPReg)
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

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

if args.dataset == "lsn":
    x, y, adj, datafile = generate_lsn(n=args.num_nodes,
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
    raise NotImplementedError("Dataset {} is not supported.".format(
        args.dataset))

m_adj = adj
if args.model_type.startswith("noisy"):
    rs = np.random.RandomState(0)
    temp = adj + rs.normal(0, 0.2, size=adj.shape)
    temp[temp > 0.5] = 1
    temp[temp <= 0.5] = 0
    temp += temp.T
    temp[temp > 0] = 1
    m_adj = temp

model_args = {
    "num_features": data.x.size(1),
    "hidden_size": args.hidden_size,
    "dropout": args.dropout,
    "activation": "relu"
}

if "spectral" in args.model_type:
    model_args["adj"] = m_adj

if args.model_type in ["mlp", "mnmlp"]:
    model = MLPReg(**model_args)
elif args.model_type in ["gcn", "mngcn"]:
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
elif args.model_type in ["cmlp", "noisycmlp"]:
    model = CMLPReg(**model_args)
elif args.model_type in ["cgcn", "noisycgcn"]:
    model = CGCNReg(**model_args)
elif args.model_type in ["newcmlp", "noisynewcmlp", "condnewcmlp"]:
    model = NewCMLPReg(**model_args)
elif args.model_type in ["newcgcn", "noisynewcgcn", "condnewcgcn"]:
    model = NewCGCNReg(**model_args)
elif args.model_type in ["spectralcmlp"]:
    model = SpectralCMLPReg(**model_args)
elif args.model_type in ["spectralcgcn"]:
    model = SpectralCGCNReg(**model_args)
elif args.model_type in ["regressioncmlp"]:
    model = RegressionCMLPReg(**model_args)
elif args.model_type in ["regressioncgcn"]:
    model = RegressionCGCNReg(**model_args)
else:
    raise NotImplementedError("Model {} is not supported.".format(
        args.model_type))
model.to(args.device)

if args.opt == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError("Optimizer {} is not supported.".format(
        args.opt))

if hasattr(model, "gen"):

    if model.post_type in ["regressioncgcn"]:

        def train_loss_fn(model, data):
            post_y_pred = model(data)
            nll_generative = model.gen.nll_generative(data, post_y_pred)
            nll_discriminative = model.post.nll_regression_copula(data)
            return args.lamda * nll_generative + nll_discriminative

    else:

        def train_loss_fn(model, data):
            post_y_pred = model(data)
            nll_generative = model.gen.nll_generative(data, post_y_pred)
            nll_discriminative = criterion(post_y_pred[data.train_mask],
                                           data.y[data.train_mask])
            return args.lamda * nll_generative + nll_discriminative

elif hasattr(model, "nll_copula"):
    L = np.diag(m_adj.sum(axis=0)) - m_adj
    cov = args.m_tau * np.linalg.inv(L + args.m_gamma * np.eye(adj.shape[0]))
    cov = torch.tensor(cov, dtype=torch.float32).to(args.device)
    cov = cov[data.train_mask, :]
    cov = cov[:, data.train_mask]

    # def train_loss_fn(model, data):  # old copula loss
    #     pred = model(data)[data.train_mask]
    #     label = data.y[data.train_mask]
    #     nll_copula = model.nll_copula(pred, label, cov)
    #     nll_q = criterion(pred, label)
    #     return args.lamda * nll_copula + nll_q

    def train_loss_fn(model, data):  # new copula loss (joint NLL)
        pred = model(data)[data.train_mask]
        label = data.y[data.train_mask]
        nll_copula = model.nll_copula(pred, label, cov)
        normal = Normal(loc=pred, scale=torch.diag(cov).pow(0.5))
        nll_q = -normal.log_prob(label)
        return nll_copula + nll_q.sum()

elif hasattr(model, "nll_spectral_copula"):

    def train_loss_fn(model, data):  # new copula loss (joint NLL)
        pred = model(data)[data.train_mask]
        label = data.y[data.train_mask]
        nll = model.nll_spectral_copula(pred, label, data.train_mask)
        return nll

elif hasattr(model, "nll_regression_copula"):

    def train_loss_fn(model, data):  # new copula loss (joint NLL)
        nll = model.nll_regression_copula(data)
        return nll

elif args.model_type.startswith("mn"):
    L = np.diag(adj.sum(axis=0)) - adj
    cov = args.m_tau * np.linalg.inv(L + args.m_gamma * np.eye(adj.shape[0]))
    cov = torch.tensor(cov, dtype=torch.float32).to(args.device)
    cov = cov[data.train_mask, :]
    cov = cov[:, data.train_mask]

    def train_loss_fn(model, data):
        pred = model(data)[data.train_mask]
        label = data.y[data.train_mask]
        mn = MultivariateNormal(pred, cov)
        return -mn.log_prob(label)

else:

    def train_loss_fn(model, data):
        return criterion(model(data)[data.train_mask], data.y[data.train_mask])


if args.test_metric == "mse":
    if args.model_type.startswith("cond"):
        assert hasattr(model, "cond_predict")

        eval_L = np.diag(adj.sum(axis=0)) - adj
        eval_cov = args.m_tau * np.linalg.inv(
            eval_L + args.m_gamma * np.eye(adj.shape[0]))
        eval_cov = torch.tensor(eval_cov, dtype=torch.float32).to(args.device)

        def test_loss_fn(logits, data, mask):  # joint NLL test metric
            eval_logits = model.cond_predict(
                data, eval_cov, data.train_mask, mask, num_samples=1000)
            return criterion(eval_logits, data.y[mask]).item()
    else:
        def test_loss_fn(logits, data, mask):  # MSE test metric
            return criterion(logits[mask], data.y[mask]).item()
elif args.test_metric == "nll":
    eval_L = np.diag(adj.sum(axis=0)) - adj
    eval_cov = args.m_tau * np.linalg.inv(eval_L +
                                          args.m_gamma * np.eye(adj.shape[0]))
    eval_cov = torch.tensor(eval_cov, dtype=torch.float32).to(args.device)

    def test_loss_fn(logits, data, mask):  # joint NLL test metric
        cov = eval_cov[mask, :]
        cov = cov[:, mask]
        pred = logits[mask]
        label = data.y[mask]
        mn = MultivariateNormal(pred, cov)
        return -mn.log_prob(label)


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
            logits = model.predict(data)
        else:
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

# rs = np.random.RandomState(0)
# temp = adj + rs.normal(0, 0.2, size=adj.shape)
# temp[temp > 0.5] = 1
# temp[temp <= 0.5] = 0
# L = np.diag(temp.sum(axis=0)) - temp
# print(np.sum(np.abs(temp - adj)))

if args.verbose == 0:
    result_path = os.path.join(args.path, "results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(
            os.path.join(
                result_path,
                ("valid__{}__test__{}__epoch__{}__model__{}__lamda__{}__"
                 "m_gamma__{}__m_tau__{}__"
                 "datafile__{}").format(selected_metrics[0],
                                        selected_metrics[1], epoch,
                                        args.model_type, args.lamda,
                                        args.m_gamma, args.m_tau,
                                        os.path.splitext(datafile)[0])),
            "w") as f:
        pass
