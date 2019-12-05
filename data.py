import os

import numpy as np
import torch
from six.moves import cPickle as pickle
from torch_geometric.data import Data


def to_data(x, y, adj, train_size=1. / 3, valid_size=1. / 3, test_size=1. / 3):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    edge_index = torch.tensor(np.array(list(adj.nonzero())))

    data = Data(x=x_tensor, y=y_tensor, edge_index=edge_index)
    n = data.x.size(0)
    if isinstance(train_size, float):
        train_size = int(n * train_size)
        valid_size = int(n * valid_size)
        test_size = int(n * test_size)
    s = train_size + valid_size + test_size
    assert s <= n, "{} > n={}".format(s, n)
    data.train_mask = torch.zeros(n).to(dtype=torch.bool)
    data.train_mask[:train_size] = True
    data.valid_mask = torch.zeros(n).to(dtype=torch.bool)
    data.valid_mask[train_size:train_size + valid_size] = True
    data.test_mask = torch.zeros(n).to(dtype=torch.bool)
    data.test_mask[-test_size:] = True
    return data


def generate_lsn(n=300,
                 d=10,
                 m=1500,
                 gamma=0.1,
                 seed=0,
                 root='./data',
                 save_file=False):
    rs = np.random.RandomState(seed=seed)
    x = rs.normal(size=(n, d))
    w_a = rs.normal(size=(d, d))
    w_y = rs.normal(size=(d, ))

    prod = x.dot(w_a)  # (n, d)
    logits = -np.linalg.norm(
        prod.reshape(1, n, d) - prod.reshape(n, 1, d), axis=2)  # (n, n)
    threshold = np.sort(logits.reshape(-1))[-m]
    adj = (logits >= threshold).astype(float)
    L = np.diag(adj.sum(axis=0)) - adj

    y_mean = np.diag(1. / adj.sum(axis=0)).dot(adj).dot(x).dot(w_y)
    y_cov = np.linalg.inv(L + gamma * np.eye(n))
    y = rs.multivariate_normal(y_mean, y_cov)

    if save_file:
        path = os.path.join(root, "lsn")
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(((x, adj, y), (w_a, w_y)),
                    open(
                        os.path.join(path, "lsn_n{}_d{}_m{}_s{}.pkl".format(
                            n, d, m, seed)), "wb"))

    return x, y, adj
