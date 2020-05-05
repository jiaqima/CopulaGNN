import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import (
    MultivariateNormal,
    _batch_mahalanobis,
)


def _standard_normal_quantile(u):
    # Ref: https://en.wikipedia.org/wiki/Normal_distribution
    return math.sqrt(2) * torch.erfinv(2 * u - 1)


def _standard_normal_cdf(x):
    # Ref: https://en.wikipedia.org/wiki/Normal_distribution
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class GaussianCopula(Distribution):
    r"""
    A Gaussian copula. 

    Args:
        covariance_matrix (torch.Tensor): positive-definite covariance matrix
    """
    arg_constraints = {"covariance_matrix": constraints.positive_definite}
    support = constraints.interval(0.0, 1.0)
    has_rsample = True

    def __init__(self, covariance_matrix=None, validate_args=None):
        # convert the covariance matrix to the correlation matrix
        # self.covariance_matrix = covariance_matrix.clone()
        # batch_diag = torch.diagonal(self.covariance_matrix, dim1=-1, dim2=-2).pow(-0.5)
        # self.covariance_matrix *= batch_diag.unsqueeze(-1)
        # self.covariance_matrix *= batch_diag.unsqueeze(-2)
        diag = torch.diag(covariance_matrix).pow(-0.5)
        self.covariance_matrix = (
            torch.diag(diag)).matmul(covariance_matrix).matmul(
            torch.diag(diag))

        batch_shape, event_shape = (
            covariance_matrix.shape[:-2],
            covariance_matrix.shape[-1:],
        )

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        self.multivariate_normal = MultivariateNormal(
            loc=torch.zeros(event_shape),
            covariance_matrix=self.covariance_matrix,
            validate_args=validate_args,
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value_x = _standard_normal_quantile(value)
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

    def conditional_sample(
        self, cond_val, sample_shape=torch.Size([]), cond_idx=None, sample_idx=None
    ):
        """
        Draw samples conditioning on cond_val.

        Args:
            cond_val (torch.Tensor): conditional values. Should be a 1D tensor.
            sample_shape (torch.Size): same as in 
                `Distribution.sample(sample_shape=torch.Size([]))`.
            cond_idx (torch.LongTensor): indices that correspond to cond_val.
                If None, use the last m dimensions, where m is the length of cond_val.
            sample_idx (torch.LongTensor): indices to sample from. If None, sample 
                from all remaining dimensions.

        Returns:
            Generates a sample_shape shaped sample or sample_shape shaped batch of 
                samples if the distribution parameters are batched.
        """
        m, n = *cond_val.shape, *self.event_shape

        if cond_idx is None:
            cond_idx = torch.arange(n - m, n)
        if sample_idx is None:
            sample_idx = torch.tensor(
                [i for i in range(n) if i not in set(cond_idx.tolist())]
            )

        assert (
            len(cond_idx) == m
            and len(sample_idx) + len(cond_idx) <= n
            and not set(cond_idx.tolist()) & set(sample_idx.tolist())
        )

        cov_00 = self.covariance_matrix.index_select(
            dim=0, index=sample_idx
        ).index_select(dim=1, index=sample_idx)
        cov_01 = self.covariance_matrix.index_select(
            dim=0, index=sample_idx
        ).index_select(dim=1, index=cond_idx)
        cov_10 = self.covariance_matrix.index_select(
            dim=0, index=cond_idx
        ).index_select(dim=1, index=sample_idx)
        cov_11 = self.covariance_matrix.index_select(
            dim=0, index=cond_idx
        ).index_select(dim=1, index=cond_idx)

        cond_val_nscale = _standard_normal_quantile(cond_val)  # Phi^{-1}(u_cond)
        reg_coeff, _ = torch.solve(cov_10, cov_11)  # Sigma_{11}^{-1} Sigma_{10}
        cond_mu = torch.mv(reg_coeff.t(), cond_val_nscale)
        cond_sigma = cov_00 - torch.mm(cov_01, reg_coeff)

        # ### direct sample
        # cond_normal = MultivariateNormal(loc=cond_mu, covariance_matrix=cond_sigma)
        # samples_nscale = cond_normal.sample(sample_shape)
        # ### direct sample

        # ### Cholesky reparameterization
        identity_mat = torch.eye(
            cond_mu.size(0), dtype=cond_mu.dtype, device=cond_mu.device)
        std_normal = MultivariateNormal(
            loc=torch.zeros_like(cond_mu), covariance_matrix=identity_mat)
        cond_sigma_cholesky = torch.cholesky(cond_sigma)
        samples_noise = std_normal.sample(sample_shape)
        samples_nscale = cond_mu.unsqueeze(0) + torch.stack(
            [torch.matmul(cond_sigma_cholesky, samples_noise[i])
             for i in range(samples_noise.size(0))])
        # ### Cholesky reparameterization

        samples_uscale = _standard_normal_cdf(samples_nscale)

        return samples_uscale


if __name__ == "__main__":
    covariance_matrix = torch.tensor(
        [
            [1.0, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.5],
            [0.5, 0.5, 0.5, 1.0],
        ]
    )
    gaussian_copula = GaussianCopula(covariance_matrix=covariance_matrix)
    cond_samples = gaussian_copula.conditional_sample(
        torch.Tensor([0.1]), sample_shape=[5]
    )
    print(cond_samples)

    from torch.distributions.normal import Normal

    covariance_matrix = torch.tensor([[1.5, 0.5], [0.5, 2.0]])
    multivariate_normal = MultivariateNormal(
        loc=torch.zeros(2), covariance_matrix=covariance_matrix
    )
    normal = Normal(loc=torch.zeros(2), scale=torch.diag(covariance_matrix).pow(0.5))
    gaussian_copula = GaussianCopula(covariance_matrix=covariance_matrix)

    for _ in range(10):
        value_x = torch.randn(5, 2)
        value_u = normal.cdf(value_x)
        actual = gaussian_copula.log_prob(value_u)
        expected = multivariate_normal.log_prob(value_x) - normal.log_prob(value_x).sum(
            -1
        )

        print(f"expected: {expected}, actual: {actual}.")
        assert torch.norm(actual - expected) < 1e-5
