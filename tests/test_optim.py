# -*- coding: utf-8 -*-
"""
    tests.optim
    ~~~~~~~~~~~

    Test the optim module.
"""
import numpy as np
import pytest
import limetr.optim as optim
import limetr.utils as utils


@pytest.mark.parametrize(("w", "h", "tr_w"),
                         [(np.ones(10), 9, np.repeat(0.9, 10)),
                          (np.ones(10), 10, np.ones(10))])
def test_proj_capped_simplex(w, h, tr_w):
    my_w = optim.proj_capped_simplex(w, h)
    assert np.linalg.norm(tr_w - my_w) < 1e-10


@pytest.fixture
def opt_solver():
    n = np.array([2, 3, 5])
    k_beta = 2
    k_gamma = k_beta
    k = k_beta + k_gamma
    m = len(n)
    N = sum(n)

    y = np.random.randn(N)
    s = np.random.rand(N)*0.09 + 0.01
    x = np.random.randn(N, k_beta)
    f = utils.create_linear_smooth_fun(x)
    z = x.copy()

    bl = np.array([-np.inf]*k_beta + [0.0]*k_gamma)
    bu = np.array([np.inf]*k)

    qm = np.array([0.0]*k)
    qw = np.array([0.01]*k)

    cf = utils.create_linear_smooth_fun(np.array([[1.0]*k_beta +
                                                  [0.0]*k_gamma]))
    cl = np.array([1.0]*cf.shape[0])
    cu = np.array([1.0]*cf.shape[0])

    pf = utils.create_linear_smooth_fun(np.array([[1.0]*k_beta +
                                                  [0.0]*k_gamma]))
    pm = np.array([1.0]*pf.shape[0])
    pw = np.array([1.0]*pf.shape[0])

    return optim.OptimizationSolver(n, y, f, z, s=s,
                                    bl=bl, bu=bu,
                                    qm=qm, qw=qw,
                                    cf=cf, cl=cl, cu=cu,
                                    pf=pf, pm=pm, pw=pw)
