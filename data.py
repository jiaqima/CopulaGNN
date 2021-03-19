import os

import numpy as np
import scipy.sparse as sp
import torch
from six.moves import cPickle as pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops


def to_data(x, y, adj=None, edge_index=None, train_idx=None, valid_idx=None,
            test_idx=None, train_size=1. / 3, valid_size=1. / 3):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    if edge_index is None:
        assert adj is not None
        edge_index = torch.tensor(np.array(list(adj.nonzero())))
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
    edge_index = remove_self_loops(to_undirected(edge_index))[0]

    data = Data(x=x_tensor, y=y_tensor, edge_index=edge_index)
    n = data.x.size(0)
    if train_idx is not None:
        assert valid_idx is not None and test_idx is not None
        all_idx = set(list(range(n)))
        train_idx = set(train_idx)
        valid_idx = set(valid_idx)
        test_idx = all_idx.difference(train_idx.union(valid_idx))
    elif isinstance(train_size, float):
        train_size = int(n * train_size)
        valid_size = int(n * valid_size)
        test_size = n - train_size - valid_size
        train_idx = set(range(train_size))
        valid_idx = set(range(train_size, train_size + valid_size))
        test_idx = set(range(n - test_size, n))
    assert len(test_idx.intersection(train_idx.union(valid_idx))) == 0
    data.train_mask = torch.zeros(n).to(dtype=torch.bool)
    data.train_mask[list(train_idx)] = True
    data.valid_mask = torch.zeros(n).to(dtype=torch.bool)
    data.valid_mask[list(valid_idx)] = True
    data.test_mask = torch.zeros(n).to(dtype=torch.bool)
    data.test_mask[list(test_idx)] = True
    return data


def generate_lsn(n=300,
                 d=10,
                 m=1500,
                 gamma=0.1,
                 tau=1.,
                 seed=1,
                 lsn_mode="xw",
                 root='./data',
                 load_file=False,
                 save_file=False):
    path = os.path.join(root, "lsn")
    assert lsn_mode in ["xw", "daxwi", "daxw"]
    filename = "lsn_{}_n{}_d{}_m{}_g{}_t{}_s{}.pkl".format(
        lsn_mode, n, d, m, gamma, tau, seed)
    if load_file and os.path.exists(os.path.join(path, filename)):
        with open(os.path.join(path, filename), "rb") as f:
            data, params = pickle.load(f)
        return data[0], data[1], data[2], filename

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

    if lsn_mode == "xw":
        y_mean = x.dot(w_y)
    else:
        y_mean = np.diag(1. / adj.sum(axis=0)).dot(adj).dot(x).dot(w_y)

    if lsn_mode == "daxwi":
        y_cov = tau * np.linalg.inv(gamma * np.eye(n))
    else:
        y_cov = tau * np.linalg.inv(L + gamma * np.eye(n))
    y = rs.multivariate_normal(y_mean, y_cov)

    if save_file:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, filename), "wb") as f:
            pickle.dump(((x, y, adj), (w_a, w_y)), f)

    return x, y, adj, filename


def read_wiki(path, name="chameleon", seed=1):
    data_path = os.path.join(path, "count")

    x = np.load(os.path.join(data_path, "wiki-{}-x.npy".format(name)))
    y = np.load(os.path.join(data_path, "wiki-{}-y.npy".format(name)))
    edge_index = np.load(
        os.path.join(data_path, "wiki-{}-edge.npy".format(name)))

    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(y))
    split_size = int(len(idx) / 3)
    train_idx = idx[:1 * split_size]
    valid_idx = idx[1 * split_size:2 * split_size]
    test_idx = idx[-split_size:]
    data = to_data(x, y, edge_index=edge_index, train_idx=train_idx,
                   valid_idx=valid_idx, test_idx=test_idx)
    return data


def read_emnlp(path, seed=1):
    data_path = os.path.join(path, "count")
    x = np.load(os.path.join(data_path, "emnlp-x.npy"))
    y = np.load(os.path.join(data_path, "emnlp-y.npy"))
    adj = sp.load_npz(os.path.join(data_path, "emnlp-adj.npz"))
    adj = adj.todense()

    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(y))
    split_size = int(len(idx) / 3)
    train_idx = idx[:1 * split_size]
    valid_idx = idx[1 * split_size:2 * split_size]
    test_idx = idx[-split_size:]
    data = to_data(x, y, adj=adj, train_idx=train_idx,
                   valid_idx=valid_idx, test_idx=test_idx)
    return data


def read_election(path, target="election", seed=1):
    data_path = os.path.join(path, "election")

    edge_file = "2012_adj.csv"
    with open(os.path.join(data_path, edge_file)) as f:
        edge_index = np.loadtxt(f, dtype=int, delimiter=",", skiprows=1)
        edge_index = edge_index.T - 1
    feature_file = "2012_xy.csv"
    with open(os.path.join(data_path, feature_file)) as f:
        x = np.loadtxt(f, dtype=float, delimiter=",", skiprows=1)
    if target == "income":
        pos = 0
    elif target == "education":
        pos = 4
    elif target == "unemployment":
        pos = 5
    elif target == "election":
        pos = 6
    else:
        NotImplementedError("Unexpected target type {}".format(target))
    y = x[:, pos]
    x = np.concatenate((x[:, :pos], x[:, pos+1:]), axis=1)

    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(y))
    split_size = int(len(idx) / 5)
    train_idx = idx[:3 * split_size]
    valid_idx = idx[3 * split_size:4 * split_size]
    test_idx = idx[-split_size:]
    data = to_data(x, y, edge_index=edge_index, train_idx=train_idx,
                   valid_idx=valid_idx, test_idx=test_idx)
    return data


if __name__ == "__main__":
    # Test to_data
    x = np.arange(6).reshape(3, 2)
    y = np.ones((3,))
    y[0] = 0
    adj = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    data = to_data(x, y, adj=adj)
    assert (data.edge_index.numpy() == np.array([[0, 1], [1, 0]])).all()
    assert (data.train_mask.numpy() == np.array([True, False, False])).all()
    assert (data.test_mask.numpy() == np.array([False, False, True])).all()
    assert (data.x.numpy() == x).all()
    assert (data.y.numpy() == y).all()

    adj = np.array([
        [1, 1, 0],
        [0, 0, 0],
        [1, 0, 0]
    ])
    data = to_data(x, y, adj=adj, train_idx=[2], valid_idx=[1], test_idx=[0])
    print(data.edge_index)
    assert (data.train_mask.numpy() == np.array([False, False, True])).all()
    assert (data.test_mask.numpy() == np.array([True, False, False])).all()

    edge_index = np.array([[1, 0], [0, 2]])
    data = to_data(x, y, edge_index=edge_index)
    assert (data.edge_index.numpy() == np.array([[0, 0, 1, 2], [1, 2, 0, 0]])).all()

    for target in ["election", "income", "education", "unemployment"]:
        print("reading ", target)
        data = read_election("data", target)
        assert len(data.x) == data.edge_index.max().item() + 1
