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

