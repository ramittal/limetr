# -*- coding: utf-8 -*-
"""
    limetr.stats
    ~~~~~~~~~~~~

    Statistical module for limetr object.
"""
import numpy as np
from . import linalg


dtype_list = [
    "gaussian",
    "uniform",
]

ptype_list = [
    "direct_prior",
    "function_prior"
]


class Prior:
    def __init__(self, ptype, dtype, dparams, fun=None):
        self.ptype = ptype
        self.dtype = dtype
        self.dparams = dparams
        self.fun = fun

        self.check()

    def check(self):
        # check distribution
        assert isinstance(self.dparams, np.ndarray)
        assert self.dparams.ndim == 2
        assert self.dparams.shape[0] == 2
        assert self.dtype in dtype_list
        if self.dtype == "gaussian":
            assert np.all(self.dparams[1] > 0.0)
        if self.dtype == "uniform":
            assert np.all(self.dparams[0] <= self.dparams[1])

        # check prior
        assert self.ptype in ptype_list
        if self.ptype == "function_prior":
            assert isinstance(self.fun, linalg.SmoothFunction)
            assert self.fun.shape[0] == self.dparams.shape[1]
