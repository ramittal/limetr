# -*- coding: utf-8 -*-
"""
    limetr.core
    ~~~~~~~~~~~

    This module implement the core classes of limetr package.
"""
import numpy as np
from . import linalg
from . import utils
from . import stats


class LimeTrVariable:
    def __init__(self, size, name="", values=None, priors=None):
        self.size = size
        if name is None:
            name = ""
        self.name = name
        self.values = values
        self.priors = priors

    def check_input(self):
        assert isinstance(self.size, int)
        assert self.size > 0
        assert isinstance(self.name, str)
        assert isinstance(self.values, np.ndarray) | self.values is None
        if self.priors:
            ok = [self.check_prior(prior) for prior in self.priors]
            assert all(ok)

    def check_prior(self, prior):
        ok = isinstance(prior, stats.Prior)
        if prior.ptype == "direct_prior":
            ok = ok and prior.dparams.shape[1] == self.size
        if prior.ptype == "function_prior":
            ok = ok and prior.fun.shape[1] == self.size
            ok = ok and prior.fun.shape[0] == prior.dparams.shape[1]
        return ok

    def add_priors(self, priors):
        if not isinstance(priors, list):
            priors = [priors]
        ok = [self.check_prior(prior) for prior in priors]
        assert all(ok)
        if self.priors is None:
            self.priors = priors
        else:
            self.priors += priors

    def set_values(self, values):
        assert isinstance(values, np.ndarray)
        assert values.size == self.size
        self.values = values


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
        self.check_input_dim()

        # variables
        dim_beta = self.fe_fun.shape[1]
        dim_gamma = self.re_mat.shape[1]
        if self.obs_sd is None:
            if self.share_obs_sd:
                dim_delta = 1
            else:
                dim_delta = self.num_groups
        else:
            dim_delta = 0

        self.variables = {
            "beta": LimeTrVariable(dim_beta,
                                   name="beta"),
            "gamma": LimeTrVariable(dim_gamma,
                                    name="gamma")
        }
        if dim_delta != 0:
            self.variables.update({
                "delta": LimeTrVariable(dim_delta,
                                        name="delta")
            })

        # create default prior
        self.variables["gamma"].add_priors([stats.Prior(
            "direct_prior",
            "uniform",
            utils.create_positive_uniform_dparams(dim_gamma),
            name="default prior"
        )])
        if "delta" in self.variables.keys():
            self.variables["delta"].add_priors([stats.Prior(
                "direct_prior",
                "uniform",
                utils.create_positive_uniform_dparams(dim_delta),
                name="default prior"
            )])

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
