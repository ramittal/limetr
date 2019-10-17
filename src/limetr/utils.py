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


def create_positive_uniform_dparams(size):
    assert isinstance(size, int)
    assert size >= 0
    dparams = np.array([[0.0]*size,
                        [np.inf]*size])
    return dparams


def create_negative_uniform_dparams(size):
    assert isinstance(size, int)
    assert size >= 0
    dparams = np.array([[-np.inf]*size,
                        [0.0]*size])
    return dparams
