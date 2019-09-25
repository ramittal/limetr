# -*- coding: utf-8 -*-
"""
    limetr.distr
    ~~~~~~~~~~~~

    Distribution class for limetr object.
"""
import numpy as np


distr_list = [
    "Gaussian",
    "Uniform",
]


class Gaussian:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd
        self.size = mean.size

    def check(self):
        assert type(self.mean) == np.ndarray
        assert type(self.sd) == np.ndarray
        assert self.mean.size == self.sd.size
        assert self.mean.ndim == 1
        assert self.sd.ndim == 1
        assert np.all(self.sd >= 0.0)


class Uniform:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        self.size = lb.size

    def check(self):
        assert type(self.lb) == np.ndarray
        assert type(self.ub) == np.ndarray
        assert self.lb.size == self.ub.size
        assert np.all(self.lb <= self.ub)


class Prior:
    def __init__(self, distr, fun=None):
        self.distr = distr
        self.fun = fun
        if self.fun is None:
            self.prior_type = "direct_prior"
        else:
            self.prior_type = "function_prior"

    def check(self):
        if self.prior_type == "function_prior":
            assert self.fun.shape[0] == self.distr.size
