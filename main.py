from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import to_dense_adj

from data import load_data
from models import (GAT, GCN, GenGNN, MLP,
                    RegressionCGCN, SpectralCGCN)

from models import _one_hot

# Training settings
parser = argparse.ArgumentParser()

# General configs.
parser.add_argument("--dataset", default="cora")
parser.add_argument("--model", default="gcn")
parser.add_argument("--num_labels_per_class", type=int, default=20)
parser.add_argument("--result_path", default="results")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    '--missing_edge',
    action='store_true',
    default=False,
    help='Missing edge in test set.')
parser.add_argument(
    "--epochs", type=int, default=2000, help="Number of epochs to train.")
parser.add_argument(
    "--patience", type=int, default=200, help="Early stopping patience.")
parser.add_argument("--device", default="cuda")
parser.add_argument("--verbose", type=int, default=1, help="Verbose.")
parser.add_argument(
    '--eval_cov',
    action='store_true',
    default=False,
    help='Whether evaluate the cov matrix.')
parser.add_argument(
    '--stop_acc',
    action='store_true',
    default=False,
    help='Whether early stop by acc.')

# Common hyper-parameters.
parser.add_argument(
    "--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).")
parser.add_argument(
    "--hidden", type=int, default=64, help="Number of hidden units.")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    help="Dropout rate (1 - keep probability).")
parser.add_argument("--activation", default="relu")
parser.add_argument(
    "--temperature", type=float, default=1.0, help="Softmax temperature.")

# GAT hyper-parameters.
parser.add_argument(
    "--num_heads", type=int, default=8, help="Number of heads.")

# Generative model hyper-parameters.
parser.add_argument(
    "--lamda",
    type=float,
    default=1.0,
    help="Lambda coefficient for nll_discriminative.")
parser.add_argument(
    "--neg_ratio", type=float, default=1.0, help="Negative sample ratio.")

# LSM hyper-parameters.
parser.add_argument(
    "--hidden_x",
    type=int,
    default=2,
    help="Number of hidden units for x_enc.")

# SBM hyper-parameters.
parser.add_argument("--p0", type=float, default=0.9, help="p0 in SBM.")
parser.add_argument("--p1", type=float, default=0.1, help="p1 in SBM.")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

data = load_data(
    dataset=args.dataset).to(args.device)
adj = to_dense_adj(data.edge_index).cpu().numpy()

model_args = {
    "num_features": data.num_features,
    "num_classes": data.num_classes,
    "hidden_size": args.hidden,
    "dropout": args.dropout,
    "activation": args.activation,
    "temperature": args.temperature
}

if args.model == "gcn":
    model = GCN(**model_args)
elif args.model == "gat":
    model_args["num_heads"] = args.num_heads
    model_args["hidden_size"] = int(args.hidden / args.num_heads)
    model = GAT(**model_args)
elif args.model == "mlp":
    model = MLP(**model_args)
elif args.model == "spectralcgcn":
    model_args["adj"] = adj
    model = SpectralCGCN(**model_args)
elif args.model == "regressioncgcn":
    model = RegressionCGCN(**model_args)
else:
    gen_type, post_type = args.model.split("_")

    gen_config = copy.deepcopy(model_args)
    gen_config["type"] = gen_type
    gen_config["neg_ratio"] = args.neg_ratio
    if gen_type == "lsm":
        gen_config["hidden_x"] = args.hidden_x
    if gen_type == "sbm":
        gen_config["p0"] = args.p0
        gen_config["p1"] = args.p1

    post_config = copy.deepcopy(model_args)
    post_config["type"] = post_type
    if post_type == "gat":
        post_config["num_heads"] = args.num_heads
        post_config["hidden_size"] = int(args.hidden / args.num_heads)
    if post_type == "spectralcgcn":
        post_config["adj"] = adj
    model = GenGNN(gen_config, post_config)

model = model.to(args.device)
optimizer = optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

criterion = nn.CrossEntropyLoss()

if hasattr(model, "gen"):

    if model.post_type in ["regressioncgcn"]:

        def train_loss_fn(model, data):
            post_y_pred = model(data)
            nll_generative = model.gen.nll_generative(data, post_y_pred)
            nll_discriminative = model.post.nll(data)
            return args.lamda * nll_generative + nll_discriminative

    else:

        def train_loss_fn(model, data):
            post_y_log_prob = model(data)
            nll_generative = model.gen.nll_generative(data, post_y_log_prob)
            nll_discriminative = criterion(post_y_log_prob[data.train_mask],
                                           data.y[data.train_mask])
            return nll_generative + args.lamda * nll_discriminative

elif hasattr(model, "nll"):

    def train_loss_fn(model, data):
        return model.nll(data)

else:

    def train_loss_fn(model, data):
        return criterion(
            model(data)[data.train_mask], data.y[data.train_mask])


if args.stop_acc:

    def val_loss_fn(logits, data):
        pred = logits[data.val_mask].max(1)[1]
        acc = pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        return -acc

elif hasattr(model, "predict") or (hasattr(model, "post") and model.post_type in ["regressioncgcn"]):

    def val_loss_fn(logits, data):
        return F.nll_loss(torch.log(logits[data.val_mask]), data.y[data.val_mask]).item()

else:

    def val_loss_fn(logits, data):
        return criterion(logits[data.val_mask], data.y[data.val_mask]).item()


def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


def eval_cov():
    if hasattr(model, "post"):
        cov = model.post.get_cov(data)
    else:
        cov = model.get_cov(data)
    diag = torch.diag(cov).pow(-0.5)
    corr = (torch.diag(diag)).matmul(cov).matmul(torch.diag(diag))
    cov = cov * (1 - eye_like(cov))
    cov[data.train_mask] = 0
    corr = corr * (1 - eye_like(corr))
    corr[data.train_mask] = 0
    cov_sort = cov.reshape(-1).sort(descending=True)[0]
    corr_sort = corr.reshape(-1).sort(descending=True)[0]
    k_list = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    prec_cov = []
    prec_corr = []
    prec_cov_rnd = []
    prec_corr_rnd = []
    for k in k_list:
        cov_thresh = cov_sort[k]
        cov_idx = torch.nonzero(cov > cov_thresh)
        cov_rnd = torch.randperm(cov_idx.size(0), device=cov_idx.device)
        corr_thresh = corr_sort[k]
        corr_idx = torch.nonzero(corr > corr_thresh)
        corr_rnd = torch.randperm(corr_idx.size(0), device=corr_idx.device)
        prec_cov.append(
            torch.mean((data.y[cov_idx[:, 0]] == data.y[cov_idx[:, 1]]).to(
                dtype=torch.float32)).item())
        prec_cov_rnd.append(
            torch.mean((data.y[cov_idx[cov_rnd, 0]] == data.y[cov_idx[:, 1]]).to(
                dtype=torch.float32)).item())
        prec_corr.append(
            torch.mean((data.y[corr_idx[:, 0]] == data.y[corr_idx[:, 1]]).to(
                dtype=torch.float32)).item())
        prec_corr_rnd.append(
            torch.mean((data.y[corr_idx[corr_rnd,
                                        0]] == data.y[corr_idx[:, 1]]).to(
                                            dtype=torch.float32)).item())

    print("prec @    {}".format(", ".join(["%.1e" % k for k in k_list])))
    print("cov:      {}".format(", ".join(
        ["%.5f" % prec for prec in prec_cov])))
    print("cov rnd:  {}".format(", ".join(["%.5f" % prec
                                           for prec in prec_cov_rnd])))
    print("corr:     {}".format(", ".join(
        ["%.5f" % prec for prec in prec_corr])))
    print("corr rnd: {}".format(", ".join(
        ["%.5f" % prec for prec in prec_corr_rnd])))


def train():
    model.train()
    optimizer.zero_grad()
    loss = train_loss_fn(model, data)
    loss.backward()
    optimizer.step()


def test():
    model.eval()
    if hasattr(model, "predict"):
        logits = model.predict(data, num_samples=1000)
    elif hasattr(model, "post") and model.post_type in ["regressioncgcn"]:
        logits = model(data, num_samples=1000)
    else:
        logits = model(data)
    val_loss = val_loss_fn(logits, data)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return val_loss, accs


if args.eval_cov:
    if not hasattr(model, "get_cov"):
        if not hasattr(model, "post"):
            args.eval_cov = False
        elif not hasattr(model.post, "get_cov"):
            args.eval_cov = False

if args.eval_cov:
    eval_cov()
# Training.
patience = args.patience
best_val_loss = np.inf
selected_accs = None
for epoch in range(1, args.epochs):
    if patience < 0:
        break
    train()
    val_loss, accs = test()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        selected_accs = accs
        patience = args.patience
        if args.verbose > 0:
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, *accs))
            if args.eval_cov:
                eval_cov()
    patience -= 1

# Save results.
if args.verbose < 1:
    result_path = os.path.join(
        args.result_path,
        "%s/nl%d" % (args.dataset, args.num_labels_per_class))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results = "vacc_%.4f_tacc_%.4f_seed_%d" % (
        selected_accs[1], selected_accs[2], args.seed)
    model_settng = "model_%s_lr_%.6f_h_%03d_l2_%.6f" % (
        args.model, args.lr, args.hidden, args.weight_decay)
    misc_hp = "act_%s_nh_%d_lambda_%.2f_nr_%.2f_hx_%d_p0_%.2f_p1_%.2f_do_%.2f" % (
        args.activation, args.num_heads, args.lamda, args.neg_ratio,
        args.hidden_x, args.p0, args.p1, args.dropout)
    fname = os.path.join(result_path,
                         "_".join([results, model_settng, misc_hp]))
    with open(fname, "w") as f:
        pass
