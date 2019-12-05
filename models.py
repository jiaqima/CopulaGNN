from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GCNReg(nn.Module):
    """GCN Regressor."""

    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.,
                 activation="relu"):
        """Initializes a GCN Regressor.
        """
        super(GCNReg, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.view(-1)


class GATReg(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.,
                 activation="relu",
                 num_heads=8):
        super(GATReg, self).__init__()
        self.conv1 = GATConv(
            num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_size * num_heads, 1, dropout=dropout)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.view(-1)


class MLPReg(torch.nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.5,
                 activation="relu"):
        super(MLPReg, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x.view(-1)


class LSMReg(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 hidden_x,
                 dropout=0.5,
                 activation="relu",
                 neg_ratio=1.0):
        super(LSMReg, self).__init__()
        self.p_y_x = MLPReg(num_features, hidden_size, dropout, activation)
        self.x_enc = nn.Linear(num_features, hidden_x)
        self.p_e_xy = nn.Linear(2 * (hidden_x + 1), 1)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.neg_ratio = neg_ratio

    def forward(self, data):
        y_mu = self.p_y_x(data)
        y_mu = torch.where(data.train_mask, data.y, y_mu).unsqueeze(1)
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = self.activation(self.x_enc(x))

        # Positive edges.
        y_query = F.embedding(data.edge_index[0], y_mu)
        y_key = F.embedding(data.edge_index[1], y_mu)
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
            y_query = F.embedding(edge_index_neg[0], y_mu)
            y_key = F.embedding(edge_index_neg[1], y_mu)
            x_query = F.embedding(edge_index_neg[0], x)
            x_key = F.embedding(edge_index_neg[1], x)
            xy = torch.cat([x_query, x_key, y_query, y_key], dim=1)
            e_pred_neg = self.p_e_xy(xy)

        return e_pred_pos, e_pred_neg, y_mu.squeeze()

    def nll_generative(self, data, post_y_mu):
        e_pred_pos, e_pred_neg, y_mu = self.forward(data)
        # unlabel_mask = data.val_mask + data.test_mask
        unlabel_mask = torch.logical_xor(
            torch.ones_like(data.train_mask).to(dtype=torch.bool),
            data.train_mask)

        # nll of p_g_xy
        nll_p_g_xy = -torch.mean(F.logsigmoid(e_pred_pos))
        if e_pred_neg is not None:
            nll_p_g_xy += -torch.mean(F.logsigmoid(-e_pred_neg))

        # nll of p_y_x
        nll_p_y_x = F.mse_loss(y_mu[data.train_mask], data.y[data.train_mask])
        nll_p_y_x += F.mse_loss(post_y_mu[unlabel_mask], y_mu[unlabel_mask])

        # nll of q_y_xg
        nll_q_y_xg = -0.5 * torch.sum(post_y_mu[unlabel_mask].pow(2))

        return nll_p_g_xy + nll_p_y_x + nll_q_y_xg


class GenGNN(nn.Module):
    def __init__(self, gen_config, post_config):
        super(GenGNN, self).__init__()
        self.gen_type = gen_config.pop("type")
        if self.gen_type == "lsm":
            self.gen = LSMReg(**gen_config)
        else:
            raise NotImplementedError(
                "Generative model type %s not supported." % self.gen_type)

        self.post_type = post_config.pop("type")
        if self.post_type == "gcn":
            self.post = GCNReg(**post_config)
        elif self.post_type == "gat":
            self.post = GATReg(**post_config)
        else:
            raise NotImplementedError(
                "Posterior model type %s not supported." % self.post_type)

    def forward(self, data):
        return self.post(data)
