from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from models import MLP
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import SEP

parser = argparse.ArgumentParser(description='Main.')
parser.add_argument("--dataset", default="lsn")
parser.add_argument("--model_type", default="mlp")
parser.add_argument("--path", default="./data/lsn")
parser.add_argument("--verbose", type=int, default=2)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=-1)

# Dataset configuration
# TODO: Add dataset specific arguments here.

# Model configuration.
# TODO: Add model specific arguments here.

# Training configuration.
# TODO: Add training specific arguments here.

# Other configuration
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--log_interval", type=int, default=500)
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

# Prepare the dataset.
dataloaders = {}
input_shape = None
if args.dataset == "lsn":
    assert args.path.strip("/").split("/")[-1] == "lsn"
    # Prepare dataloaders.
    for phase in ["train", "valid", "test"]:
        dataloaders[phase] = None
        # TODO: Implement dataloaders.
        raise NotImplementedError("Dataloaders are not implemented.")
else:
    raise NotImplementedError("Dataset %s is not supported." % args.dataset)

# Initialize the model.
model_config = {}
# TODO: Specify model config.

if args.model_type == "mlp":
    model = MLP(model_config=model_config)
    # TODO: Check if the model class instantiation is correct.
    raise NotImplementedError("ModelClass is not checked.")
else:
    raise NotImplementedError("Model %s is not supported." % args.model_type)
model.to(args.device)

# Optimization criterions.
criterions = {}
# TODO: Specify criterions.
raise NotImplementedError("`criterions` is not specified.")

# Optimizer.
train_config = {}
# TODO: Specify train config.

if train_config["optim"] == "SGD":
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"])
elif train_config["optim"] == "Adam":
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"])
elif train_config["optim"] == "SGD-M":
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        momentum=train_config["momentum"],
        nesterov=True)
elif train_config["optim"] == "Adagrad":
    optimizer = optim.Adagrad(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"])
elif train_config["optim"] == "RMSprop":
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"])
else:
    raise NotImplementedError(
        "Optimizer %s is not supported." % train_config["optim"])

# Evaluation metrics.
metrics = {}
# TODO: Specify metrics.
raise NotImplementedError("`metrics` is not specified.")
