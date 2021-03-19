from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, APPNP
from torch_geometric.utils import get_laplacian, to_dense_adj

from scipy.special import gammainc
from scipy.stats import poisson
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
from copula import GaussianCopula

EPS = 1e-3


def grad_x_gammainc(a, x, grad_output):
    temp = -x + (a-1)*torch.log(x) - torch.lgamma(a)
    temp = torch.where(temp > -25, torch.exp(temp), torch.zeros_like(temp))  # avoid underflow
    return temp * grad_output  # everything is element-wise


class GammaIncFunc(Function):
    '''Regularized lower incomplete gamma function.'''
    @staticmethod
    def forward(ctx, a, x):
        # detach so we can cast to NumPy
        a = a.detach()
        x = x.detach()
        result = gammainc(a.cpu().numpy(), x.cpu().numpy())
        result = torch.as_tensor(result, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(a, x, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        a, x, result = ctx.saved_tensors
        grad_a = torch.zeros_like(a)  # grad_a is never needed
        grad_x = grad_x_gammainc(a, x, grad_output)
        return grad_a, grad_x


def _batch_normal_icdf(loc, scale, value):
    return loc[None, :] + scale[None, :] * torch.erfinv(2 * value - 1) * math.sqrt(2)


class MLP(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = Linear(num_features, hidden_size)
        self.fc2 = Linear(hidden_size, 1)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x = data.x
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x.view(-1)


class GCN(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.view(-1)


class SAGE(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if hidden_size % 2 == 1:
            hidden_size += 1
        self.conv1 = GCNConv(num_features, hidden_size // 2)
        self.conv2 = GCNConv(hidden_size, 1)
        self.fc1 = Linear(num_features, hidden_size // 2)
        self.fc2 = Linear(hidden_size, 1)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.cat([self.conv1(x, edge_index), self.fc1(x)], dim=1)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index) + self.fc2(x)
        return x.view(-1)


class GAT(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 num_heads=8,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = GATConv(
            num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_size * num_heads, 1, dropout=dropout)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.view(-1)


class APPNPNet(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 K=10,
                 alpha=0.1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lin1 = Linear(num_features, hidden_size)
        self.lin2 = Linear(hidden_size, 1)
        self.prop1 = APPNP(K, alpha)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x.view(-1)


class CopulaModel(nn.Module):

    def __init__(self, marginal_type, eps=EPS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.marginal_type = marginal_type
        self.eps = eps

    def marginal(self, logits, cov=None):
        if self.marginal_type == "Normal":
            return Normal(loc=logits, scale=torch.diag(cov).pow(0.5))
        elif self.marginal_type == "Poisson":
            return Poisson(rate=torch.exp(logits) + self.eps)
        else:
            raise NotImplementedError(
                "Marginal type `{}` not supported.".format(self.marginal_type))

    def cdf(self, marginal, labels):
        if self.marginal_type == "Normal":
            res = marginal.cdf(labels)
        elif self.marginal_type == "Poisson":
            cdf_left = 1 - GammaIncFunc.apply(labels, marginal.mean)
            cdf_right = 1 - GammaIncFunc.apply(labels + 1, marginal.mean)
            res = (cdf_left + cdf_right) / 2
        else:
            raise NotImplementedError(
                "Marginal type `{}` not supported.".format(self.marginal_type))
        return torch.clamp(res, self.eps, 1-self.eps)

    def icdf(self, marginal, u):
        if self.marginal_type == "Normal":
            res = _batch_normal_icdf(marginal.mean, marginal.stddev, u)
        elif self.marginal_type == "Poisson":
            mean = marginal.mean.detach().cpu().numpy()
            q = u.detach().cpu().numpy()
            res = poisson.ppf(q, mean)
            res = torch.as_tensor(res, dtype=u.dtype, device=u.device)
        else:
            raise NotImplementedError(
                "Marginal type `{}` not supported.".format(self.marginal_type))
        if (res == float("inf")).sum() + (res == float("nan")).sum() > 0:
            # remove the rows containing inf or nan values
            inf_mask = res.sum(dim=-1) == float("inf")
            nan_mask = res.sum(dim=-1) == float("nan")
            res[inf_mask] = 0
            res[nan_mask] = 0
            valid_num = inf_mask.size(0) - inf_mask.sum() - nan_mask.sum()
            return res.sum(dim=0) / valid_num
        return res.mean(dim=0)

    def get_cov(self, data):
        raise NotImplementedError("`get_cov` not implemented.")

    def get_prec(self, data):
        raise NotImplementedError("`get_prec` not implemented.")

    def nll(self, data):
        cov = self.get_cov(data)
        cov = cov[data.train_mask, :]
        cov = cov[:, data.train_mask]
        logits = self.forward(data)[data.train_mask]
        labels = data.y[data.train_mask]

        copula = GaussianCopula(cov)
        marginal = self.marginal(logits, cov)

        u = self.cdf(marginal, labels)
        nll_copula = - copula.log_prob(u)
        nll_marginal = - torch.sum(marginal.log_prob(labels))
        return (nll_copula + nll_marginal) / labels.size(0)

    def predict(self, data, num_samples=500):
        cond_mask = data.train_mask
        eval_mask = torch.logical_xor(
            torch.ones_like(data.train_mask).to(dtype=torch.bool),
            data.train_mask)
        cov = self.get_cov(data)
        logits = self.forward(data)
        copula = GaussianCopula(cov)

        cond_cov = (cov[cond_mask, :])[:, cond_mask]
        cond_marginal = self.marginal(logits[cond_mask], cond_cov)
        eval_cov = (cov[eval_mask, :])[:, eval_mask]
        eval_marginal = self.marginal(logits[eval_mask], eval_cov)

        cond_u = torch.clamp(
            self.cdf(cond_marginal, data.y[cond_mask]), self.eps, 1-self.eps)
        cond_idx = torch.where(cond_mask)[0]
        sample_idx = torch.where(eval_mask)[0]
        eval_u = copula.conditional_sample(
            cond_val=cond_u, sample_shape=[num_samples, ], cond_idx=cond_idx,
            sample_idx=sample_idx)
        eval_u = torch.clamp(eval_u, self.eps, 1-self.eps)
        eval_y = self.icdf(eval_marginal, eval_u)

        pred_y = data.y.clone()
        pred_y[eval_mask] = eval_y
        return pred_y


class CorCopulaGCN(GCN, CopulaModel):

    def __init__(self,
                 num_features,
                 hidden_size,
                 marginal_type="Normal",
                 dropout=0.,
                 activation="relu"):
        super().__init__(
            num_features=num_features, hidden_size=hidden_size,
            dropout=dropout, activation=activation,
            marginal_type=marginal_type)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(3.0))
        self.S = None
        self.I = None

    def get_prec(self, data):
        if self.S is None:
            adj = to_dense_adj(data.edge_index)[0].cpu().numpy()
            degree = adj.sum(axis=0)
            degree[degree==0] = 1
            D = np.diag(degree**(-0.5))
            S = D.dot(adj).dot(D)
            self.S = torch.tensor(S, dtype=torch.float32).to(data.x.device)
            self.I = torch.eye(self.S.size(0)).to(data.x.device)
        prec = torch.exp(self.beta) * (self.I - torch.tanh(self.alpha) * self.S)
        return prec

    def get_cov(self, data):
        return torch.inverse(self.get_prec(data))


class RegCopulaGCN(GCN, CopulaModel):

    def __init__(self,
                 num_features,
                 hidden_size,
                 marginal_type="Normal",
                 dropout=0.,
                 activation="relu"):
        super().__init__(
            num_features=num_features, hidden_size=hidden_size,
            dropout=dropout, activation=activation,
            marginal_type=marginal_type)

        self.reg_fc1 = nn.Linear(num_features * 2, hidden_size)
        self.reg_fc2 = nn.Linear(hidden_size, 1)

    def get_prec(self, data):
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
        return L + torch.eye(L.size(0), dtype=L.dtype, device=L.device)

    def get_cov(self, data):
        return torch.inverse(self.get_prec(data))


class CorCopulaSAGE(SAGE, CopulaModel):

    def __init__(self,
                 num_features,
                 hidden_size,
                 marginal_type="Normal",
                 dropout=0.,
                 activation="relu"):
        super().__init__(
            num_features=num_features, hidden_size=hidden_size,
            dropout=dropout, activation=activation,
            marginal_type=marginal_type)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(3.0))
        self.S = None
        self.I = None

    def get_prec(self, data):
        if self.S is None:
            adj = to_dense_adj(data.edge_index)[0].cpu().numpy()
            degree = adj.sum(axis=0)
            degree[degree==0] = 1
            D = np.diag(degree**(-0.5))
            S = D.dot(adj).dot(D)
            self.S = torch.tensor(S, dtype=torch.float32).to(data.x.device)
            self.I = torch.eye(self.S.size(0)).to(data.x.device)
        prec = torch.exp(self.beta) * (self.I - torch.tanh(self.alpha) * self.S)
        return prec

    def get_cov(self, data):
        return torch.inverse(self.get_prec(data))


class RegCopulaSAGE(SAGE, CopulaModel):

    def __init__(self,
                 num_features,
                 hidden_size,
                 marginal_type="Normal",
                 dropout=0.,
                 activation="relu"):
        super().__init__(
            num_features=num_features, hidden_size=hidden_size,
            dropout=dropout, activation=activation,
            marginal_type=marginal_type)

        self.reg_fc1 = nn.Linear(num_features * 2, hidden_size)
        self.reg_fc2 = nn.Linear(hidden_size, 1)

    def get_prec(self, data):
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
        return L + torch.eye(L.size(0), dtype=L.dtype, device=L.device)

    def get_cov(self, data):
        return torch.inverse(self.get_prec(data))
