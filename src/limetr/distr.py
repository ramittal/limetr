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
        # check the input type
        assert type(mean) == np.ndarray
        assert type(sd) == np.ndarray
        # check the input size
        assert mean.size == sd.size
        assert mean.ndim == 1
        assert sd.ndim == 1
        # check standard deviation non-negative
        assert np.all(sd >= 0.0)
        # pass in data
        self.mean = mean
        self.sd = sd
        self.size = mean.size


class Uniform:
    def __init__(self, lb, ub):
        # check the input type
        assert type(lb) == np.ndarray
        assert type(ub) == np.ndarray
        # check the input size
        assert lb.size == ub.size
        # check lower bound smaller than the upper bound
        assert np.all(lb <= ub)
        # pass in data
        self.lb = lb
        self.ub = ub
        self.size = lb.size
        # compute mean and sd
        self.mean = 0.5*(self.lb + self.ub)
        self.sd = (self.ub - self.lb)/np.sqrt(12.0)
