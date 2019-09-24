# -*- coding: utf-8 -*-
"""
    limetr.utils
    ~~~~~~~~~~~~

    Utility classes and functions.
"""
import numpy as np
from . import linalg
from . import distr


def create_linear_smooth_fun(mat):
    # check input
    assert type(mat) == np.ndarray
    assert mat.ndim == 2

    def fun(x):
        return mat.dot(x)

    def jac_mat(x):
        return mat

    return linalg.SmoothFunction(mat.shape, fun, jac_mat=jac_mat)


def create_uninformative_uniform_distr(size):
    lb = np.repeat(-np.inf, size)
    ub = np.repeat(np.inf, size)
    return distr.Uniform(lb, ub)


def create_uninformative_gaussian_distr(size):
    mean = np.zeros(size)
    sd = np.repeat(np.inf, size)
    return distr.Gaussian(mean, sd)


def create_postive_uniform_distr(size):
    lb = np.zeros(size)
    ub = np.repeat(np.inf, size)
    return distr.Uniform(lb, ub)


def create_negative_uniform_distr(size):
    lb = np.repeat(-np.inf, size)
    ub = np.zeros(size)
    return distr.Uniform(lb, ub)
