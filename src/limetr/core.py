# -*- coding: utf-8 -*-
"""
    limetr.core
    ~~~~~~~~~~~

    This module implement the core classes of limetr package.
"""
import numpy as np
from . import linalg
from . import utils


class LimeTr:
    def __init__(self,
                 obs_mean,
                 fe_fun,
                 re_mat,
                 obs_sd=None,
                 share_obs_sd=True,
                 group_sizes=None,
                 inlier_pct=1.0):
        # data and settings
        self.obs_mean = obs_mean
        self.fe_fun = fe_fun
        self.re_mat = re_mat
        self.obs_sd = obs_sd
        self.share_obs_sd = share_obs_sd
        self.group_sizes = group_sizes
        self.inlier_pct = inlier_pct

        self.check_input_type()

        # dimensions
        self.num_obs = obs_mean.size
        if self.group_sizes is None:
            self.group_sizes = np.array([1]*self.num_obs)
        else:
            self.group_sizes = group_sizes
        self.num_groups = self.group_sizes.size

        self.dim_beta = self.fe_fun.shape[1]
        self.dim_gamma = self.re_mat.shape[1]
        if self.obs_sd is None:
            if self.share_obs_sd:
                self.dim_delta = 1
            else:
                self.dim_delta = self.num_groups
        else:
            self.dim_delta = 0
        self.dim_fe = self.dim_beta + self.dim_gamma + self.dim_delta
        self.dim_re = (self.num_groups, self.dim_gamma)

        self.check_input_dim()

        # specify default prior
        self.prior_types = [
            "uniform_direct_prior",
            "uniform_function_prior",
            "gaussian_direct_prior",
            "gaussian_function_prior",
        ]
        self.num_prior_types = len(self.prior_types)
        self.prior_type_index = {
            self.prior_types[i]: i
            for i in range(self.num_prior_types)
        }

        self.beta_priors = [None]*self.num_prior_types
        self.gamma_priors = [
            utils.create_positive_uniform_direct_prior(self.dim_gamma),
            None, None, None]
        self.delta_priors = [
            utils.create_positive_uniform_direct_prior(self.dim_delta),
            None, None, None]

    def check_input_type(self):
        assert isinstance(self.obs_mean, np.ndarray)
        assert self.obs_mean.ndim == 1

        assert isinstance(self.fe_fun, linalg.SmoothFunction)

        assert isinstance(self.re_mat, np.ndarray)
        assert self.re_mat.ndim == 2

        if self.obs_sd is not None:
            assert isinstance(self.obs_sd, np.ndarray)
            assert self.obs_sd.ndim == 1
        if self.group_sizes is not None:
            assert isinstance(self.group_sizes, np.ndarray)
            assert self.group_sizes.ndim == 1
            assert self.group_sizes.dtype == int
        assert isinstance(self.inlier_pct, (int, float))
        self.inlier_pct = float(self.inlier_pct)
        assert 0.0 <= self.inlier_pct <= 1.0

    def check_input_dim(self):
        assert self.num_obs == self.fe_fun.shape[0]
        assert self.num_obs == self.re_mat.shape[0]
        assert self.num_obs == np.sum(self.group_sizes)
