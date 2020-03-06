import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import (
    MultivariateNormal,
    _batch_mahalanobis,
)


class GaussianCopula(Distribution):
    r"""
    A Gaussian copula. 

    Args:
        covariance_matrix (Tensor): positive-definite covariance matrix
    """
    arg_constraints = {"covariance_matrix": constraints.positive_definite}
    support = constraints.interval(0.0, 1.0)
    has_rsample = True

    def __init__(self, covariance_matrix=None, validate_args=None):
        # convert the covariance matrix to the correlation matrix
        batch_diag = torch.diagonal(covariance_matrix, dim1=-1, dim2=-2).pow(-0.5)
        covariance_matrix *= batch_diag.unsqueeze(-1)
        covariance_matrix *= batch_diag.unsqueeze(-2)

        self.covariance_matrix = covariance_matrix

        batch_shape, event_shape = (
            covariance_matrix.shape[:-2],
            covariance_matrix.shape[-1:],
        )

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        self.multivariate_normal = MultivariateNormal(
            loc=torch.zeros(event_shape),
            covariance_matrix=covariance_matrix,
            validate_args=validate_args,
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # Ref: https://en.wikipedia.org/wiki/Normal_distribution
        value_x = math.sqrt(2) * torch.erfinv(2 * value - 1)
        half_log_det = (
            self.multivariate_normal._unbroadcasted_scale_tril.diagonal(
                dim1=-2, dim2=-1
            )
            .log()
            .sum(-1)
        )
        M = _batch_mahalanobis(
            self.multivariate_normal._unbroadcasted_scale_tril, value_x
        )
        M -= value_x.pow(2).sum(-1)
        return -0.5 * M - half_log_det


if __name__ == "__main__":
    from torch.distributions.normal import Normal

    covariance_matrix = torch.tensor([[1.5, 0.5], [0.5, 2.0]])
    value_u = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    gaussian_copula = GaussianCopula(covariance_matrix=covariance_matrix)
    actual = gaussian_copula.log_prob(value_u)

    normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    multivariate_normal = MultivariateNormal(
        loc=torch.zeros(2), covariance_matrix=covariance_matrix
    )
    value_x = normal.icdf(value_u)
    expected = multivariate_normal.log_prob(value_x) - normal.log_prob(value_x).sum(-1)

    print(f"expected: {expected}, actual: {actual}.")
    assert torch.norm(actual - expected) < 1e-5
