# -*- coding: utf-8 -*-
"""
    tests.distr
    ~~~~~~~~~~~

    Test the distr module.
"""
from limetr import stats
from limetr import linalg
import numpy as np
import pytest


# settings
size = 5
shape = (5, 3)
mat = np.random.randn(*shape)
smooth_fun = linalg.SmoothFunction(shape,
                                  lambda x: mat.dot(x),
                                  lambda y: mat.T.dot(y))


def test_gaussian_distr():
    mean = np.zeros(size)
    sd = np.ones(size)
    g_distr = stats.Gaussian(mean, sd)
    g_distr.check()


def test_uniform_distr():
    lb = np.zeros(size)
    ub = np.ones(size)
    u_distr = stats.Uniform(lb, ub)
    u_distr.check()


@pytest.mark.parametrize("distr",
                         [stats.Gaussian(np.zeros(size), np.ones(size)),
                          stats.Uniform(np.zeros(size), np.ones(size))])
@pytest.mark.parametrize(("fun", "prior_type"),
                         [(None, "direct_prior"),
                          (smooth_fun, "function_prior")])
def test_prior(distr, fun, prior_type):
    prior = stats.Prior(distr, fun)
    prior.check()
    assert prior.prior_type == prior_type
