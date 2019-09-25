# -*- coding: utf-8 -*-
"""
    limetr.utils
    ~~~~~~~~~~~~

    Utility classes and functions.
"""
import numpy as np
from . import linalg
from . import stats


def create_linear_smooth_fun(mat):
    # check input
    assert type(mat) == np.ndarray
    assert mat.ndim == 2

    return linalg.SmoothFunction(mat.shape,
                                 lambda x: mat.dot(x),
                                 jac_mat=lambda x: mat)


def create_dummy_uniform_distr(size):
    lb = np.repeat(-np.inf, size)
    ub = np.repeat(np.inf, size)
    return stats.Uniform(lb, ub)


def create_dummy_gaussian_distr(size):
    mean = np.zeros(size)
    sd = np.repeat(np.inf, size)
    return stats.Gaussian(mean, sd)


def create_positive_uniform_distr(size):
    lb = np.zeros(size)
    ub = np.repeat(np.inf, size)
    return stats.Uniform(lb, ub)


def create_negative_uniform_distr(size):
    lb = np.repeat(-np.inf, size)
    ub = np.zeros(size)
    return stats.Uniform(lb, ub)


def create_positive_uniform_direct_prior(size):
    udistr = create_positive_uniform_distr(size)
    return stats.Prior(udistr, fun=None)


def create_negative_uniform_direct_prior(size):
    udistr = create_negative_uniform_distr(size)
    return stats.Prior(udistr, fun=None)

