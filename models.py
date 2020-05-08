from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import get_laplacian, to_dense_adj

from copula import GaussianCopula


def batched_index_select(input, dim, index):
    if dim == -1:
        dim = len(input.shape) - 1
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def _one_hot(idx, num_class):
    return torch.zeros(len(idx), num_class).to(idx.device).scatter_(
        1, idx.unsqueeze(1), 1.)


class MLP(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 temperature=1.0):
        super(MLP, self).__init__()
        self.fc1 = Linear(num_features, hidden_size)
        self.fc2 = Linear(hidden_size, num_classes)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.temperature = temperature

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x / self.temperature


class GCN(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 temperature=1.0):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.temperature = temperature

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x / self.temperature


class GAT(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 num_heads=8,
                 temperature=1.0):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_size * num_heads, num_classes, dropout=dropout)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.temperature = temperature

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x / self.temperature


class CopulaModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="sum")

    def cdf(self, logits, labels, deterministic=True):
        deterministic = True
        probs = F.softmax(logits, dim=-1)
        boundaries = torch.cat(
            [torch.zeros(
                probs.size(0), 1, dtype=probs.dtype, device=probs.device),
             torch.cumsum(probs, dim=-1)], dim=-1)
        left = batched_index_select(boundaries, dim=-1, index=labels).view(-1)
        right = batched_index_select(boundaries, dim=-1, index=labels + 1).view(-1)
        if self.training and not deterministic:
            return (
                left + (right - left) * torch.rand_like(
                    labels.to(dtype=torch.float32)))
        else:
            return (right + left) / 2

    def icdf(self, logits, u):
        probs = F.softmax(logits, dim=-1)
        boundaries = torch.cumsum(probs, dim=-1)
        boundaries[:, -1] = 1.1  # avoid numerical error
        u = u.unsqueeze(-1)  # broadcast from the sampled prob to boundaries
        if len(u.shape) > len(boundaries.shape):
            for _ in range(len(u.shape) - len(boundaries.shape)):
                boundaries = boundaries.unsqueeze(0)
        return (u > boundaries).sum(dim=-1)

    def get_cov(self, data):
        raise NotImplementedError("`get_cov` not implemented.")

    def nll(self, data):
        cov = self.get_cov(data)
        cov = cov[data.train_mask, :]
        cov = cov[:, data.train_mask]
        logits = self.forward(data)[data.train_mask]
        labels = data.y[data.train_mask]

        n_copula = GaussianCopula(cov)
        u = self.cdf(logits, labels, deterministic=False)

        nll_q = self.ce(logits, labels)
        return (-n_copula.log_prob(u) + nll_q) / labels.size(0)

    def predict(self, data, num_samples=100):
        cond_mask = data.train_mask
        eval_mask = torch.logical_xor(
            torch.ones_like(data.train_mask).to(dtype=torch.bool),
            data.train_mask)
        cov = self.get_cov(data)
        n_copula = GaussianCopula(cov)
        logits = self.forward(data)

        cond_u = self.cdf(logits[cond_mask], data.y[cond_mask],
                          deterministic=True)
        cond_idx = torch.where(cond_mask)[0]
        sample_idx = torch.where(eval_mask)[0]
        eval_u = n_copula.conditional_sample(
            cond_val=cond_u, sample_shape=[num_samples, ], cond_idx=cond_idx,
            sample_idx=sample_idx)
        eval_y = self.icdf(logits[eval_mask], eval_u)
        # eval_y = _one_hot(eval_y, logits.size(-1))
        eval_y = _one_hot(eval_y.view(-1), logits.size(-1)).view(eval_u.size(0), eval_u.size(1), logits.size(-1))
        eval_y = eval_y.mean(dim=0)

        pred_y = _one_hot(data.y.clone(), logits.size(-1))
        pred_y[eval_mask] = eval_y
        return pred_y


class RegressionCGCN(GCN, CopulaModel):

    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.,
                 activation="relu",
                 temperature=1.0):
        super().__init__(num_features=num_features, num_classes=num_classes,
                         hidden_size=hidden_size, dropout=dropout,
                         activation=activation, temperature=temperature)

        self.reg_fc1 = nn.Linear(num_features * 2, hidden_size)
        self.reg_fc2 = nn.Linear(hidden_size, 1)

    def get_cov(self, data):
        triangle_mask = data.edge_index[0] < data.edge_index[1]
        edge_index = torch.stack([data.edge_index[0][triangle_mask],
                                  data.edge_index[1][triangle_mask]])
        x_query = F.embedding(edge_index[0], data.x)
        x_key = F.embedding(edge_index[1], data.x)
        x = torch.cat([x_query, x_key], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.reg_fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.softplus(self.reg_fc2(x))
        und_edge_index = torch.stack(
            [torch.cat([edge_index[0], edge_index[1]], dim=0),
             torch.cat([edge_index[1], edge_index[0]], dim=0)])
        und_edge_weight = torch.cat([x.view(-1), x.view(-1)], dim=0)
        L_edge_index, L_edge_weight = get_laplacian(
            und_edge_index, edge_weight=und_edge_weight,
            num_nodes=data.x.size(0))
        L = to_dense_adj(L_edge_index, edge_attr=L_edge_weight)[0]
        return torch.inverse(
            L + torch.eye(L.size(0), dtype=L.dtype, device=L.device))


class SpectralCGCN(GCN, CopulaModel):

    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 adj,
                 dropout=0.,
                 activation="relu",
                 temperature=1.0):
        super().__init__(num_features=num_features, num_classes=num_classes,
                         hidden_size=hidden_size, dropout=dropout,
                         activation=activation, temperature=temperature)

        L = np.diag(adj.sum(axis=0)) - adj
        w, v = np.linalg.eigh(L + np.eye(L.shape[0]))
        w = 1. / (w + 1e-6)
        self.register_buffer("evec", torch.tensor(v[0], dtype=torch.float32))
        self.inv_eval = nn.Parameter(torch.tensor(w[0], dtype=torch.float32))

    def get_cov(self, data):
        inv_eval = F.softplus(self.inv_eval)
        cov = self.evec.matmul(torch.diag(inv_eval)).matmul(self.evec.t())
        return cov


class LSM(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 hidden_x,
                 dropout=0.5,
                 activation="relu",
                 temperature=1.0,
                 neg_ratio=1.0):
        super(LSM, self).__init__()
        self.p_y_x = MLP(num_features, num_classes, hidden_size, dropout,
                         activation, temperature)
        self.x_enc = Linear(num_features, hidden_x)
        self.p_e_xy = Linear(2 * (hidden_x + num_classes), 1)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.neg_ratio = neg_ratio

    def forward(self, data):
        y_log_prob = F.log_softmax(self.p_y_x(data), dim=-1)
        y_prob = torch.exp(y_log_prob)
        y_prob = torch.where(
            data.train_mask.unsqueeze(1), _one_hot(data.y, y_prob.size(1)),
            y_prob)
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = self.activation(self.x_enc(x))

        # Positive edges.
        y_query = F.embedding(data.edge_index[0], y_prob)
        y_key = F.embedding(data.edge_index[1], y_prob)
        x_query = F.embedding(data.edge_index[0], x)
        x_key = F.embedding(data.edge_index[1], x)
        xy = torch.cat([x_query, x_key, y_query, y_key], dim=1)
        e_pred_pos = self.p_e_xy(xy)

        # Negative edges.
        e_pred_neg = None
        if self.neg_ratio > 0:
            num_edges_pos = data.edge_index.size(1)
            num_nodes = data.x.size(0)
            num_edges_neg = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = torch.randint(num_nodes,
                                           (2, num_edges_neg)).to(x.device)
            y_query = F.embedding(edge_index_neg[0], y_prob)
            y_key = F.embedding(edge_index_neg[1], y_prob)
            x_query = F.embedding(edge_index_neg[0], x)
            x_key = F.embedding(edge_index_neg[1], x)
            xy = torch.cat([x_query, x_key, y_query, y_key], dim=1)
            e_pred_neg = self.p_e_xy(xy)

        return e_pred_pos, e_pred_neg, y_log_prob

    def nll_generative(self, data, post_y_log_prob):
        post_y_log_prob = F.log_softmax(post_y_log_prob, dim=-1)
        e_pred_pos, e_pred_neg, y_log_prob = self.forward(data)
        # unlabel_mask = data.val_mask + data.test_mask
        unlabel_mask = torch.logical_xor(
            torch.ones_like(data.train_mask).to(dtype=torch.bool),
            data.train_mask)

        # nll of p_g_xy
        nll_p_g_xy = -torch.mean(F.logsigmoid(e_pred_pos))
        if e_pred_neg is not None:
            nll_p_g_xy += -torch.mean(F.logsigmoid(-e_pred_neg))

        # nll of p_y_x
        nll_p_y_x = F.nll_loss(y_log_prob[data.train_mask],
                               data.y[data.train_mask])
        nll_p_y_x += -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            y_log_prob[unlabel_mask])

        # nll of q_y_xg
        nll_q_y_xg = -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            post_y_log_prob[unlabel_mask])

        return nll_p_g_xy + nll_p_y_x + nll_q_y_xg


class SBM(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 temperature=1.0,
                 p0=0.9,
                 p1=0.1,
                 neg_ratio=1.0):
        super(SBM, self).__init__()
        self.p_y_x = MLP(num_features, num_classes, hidden_size, dropout,
                         activation, temperature)
        self.p0 = p0
        self.p1 = p1
        self.neg_ratio = neg_ratio

    def forward(self, data):
        y_log_prob = F.log_softmax(self.p_y_x(data), dim=-1)
        y_prob = torch.exp(y_log_prob)
        y_prob = torch.where(
            data.train_mask.unsqueeze(1), _one_hot(data.y, y_prob.size(1)),
            y_prob)

        # Positive edges.
        y_query_pos = F.embedding(data.edge_index[0], y_prob)
        y_key_pos = F.embedding(data.edge_index[1], y_prob)

        # Negative edges.
        y_query_neg = None
        y_key_neg = None
        if self.neg_ratio > 0:
            num_edges_pos = data.edge_index.size(1)
            num_nodes = data.x.size(0)
            num_edges_neg = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = torch.randint(num_nodes, (2, num_edges_neg)).to(
                y_prob.device)
            y_query_neg = F.embedding(edge_index_neg[0], y_prob)
            y_key_neg = F.embedding(edge_index_neg[1], y_prob)

        return y_query_pos, y_key_pos, y_query_neg, y_key_neg, y_log_prob

    def nll_generative(self, data, post_y_log_prob):
        post_y_log_prob = F.log_softmax(post_y_log_prob, dim=-1)
        (y_query_pos, y_key_pos, y_query_neg, y_key_neg,
         y_log_prob) = self.forward(data)
        # unlabel_mask = data.val_mask + data.test_mask
        unlabel_mask = torch.logical_xor(
            torch.ones_like(data.train_mask).to(dtype=torch.bool),
            data.train_mask)

        # nll of p_g_y
        nll_p_g_y = -torch.mean(y_query_pos * y_key_pos) * np.log(
            self.p0 / self.p1)
        if y_query_neg is not None:
            nll_p_g_y += -torch.mean(y_query_neg * y_key_neg) * np.log(
                (1 - self.p0) / (1 - self.p1))

        # nll of p_y_x
        nll_p_y_x = F.nll_loss(y_log_prob[data.train_mask],
                               data.y[data.train_mask])
        nll_p_y_x += -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            y_log_prob[unlabel_mask])

        # nll of q_y_xg
        nll_q_y_xg = -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            post_y_log_prob[unlabel_mask])

        return nll_p_g_y + nll_p_y_x + nll_q_y_xg


class GenGNN(torch.nn.Module):
    def __init__(self, gen_config, post_config):
        super(GenGNN, self).__init__()
        self.gen_type = gen_config.pop("type")
        if self.gen_type == "lsm":
            self.gen = LSM(**gen_config)
        elif self.gen_type == "sbm":
            self.gen = SBM(**gen_config)
        else:
            raise NotImplementedError(
                "Generative model type %s not supported." % self.gen_type)

        self.post_type = post_config.pop("type")
        if self.post_type == "gcn":
            self.post = GCN(**post_config)
        elif self.post_type == "gat":
            self.post = GAT(**post_config)
        elif self.post_type == "spectralcgcn":
            self.post = SpectralCGCN(**post_config)
        elif self.post_type == "regressioncgcn":
            self.post = RegressionCGCN(**post_config)
        else:
            raise NotImplementedError(
                "Generative model type %s not supported." % self.post_type)

    def forward(self, data, **predict_args):
        if hasattr(self.post, "predict"):
            return self.post.predict(data, **predict_args)
        return self.post(data)
