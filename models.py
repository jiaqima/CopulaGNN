from __future__ import division, print_function

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


class MLPReg(nn.Module):
    """MLP Regressor."""

    def __init__(self, num_features, hidden_size):
        """Initializes a MLP Regressor.
        """
        super(MLPReg, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(num_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)
        self.activation = F.relu

    def forward(self, data):
        outputs = self.linear_2(self.activation(self.linear_1(data.x)))
        return outputs.view(-1)
