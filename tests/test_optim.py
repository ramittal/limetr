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
import limetr.linalg as linalg


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

    inlier_pct = 0.95

    return optim.OptimizationSolver(n, y, f, z, s=s,
                                    bl=bl, bu=bu,
                                    qm=qm, qw=qw,
                                    cf=cf, cl=cl, cu=cu,
                                    pf=pf, pm=pm, pw=pw,
                                    inlier_pct=inlier_pct)


def true_objective(x, opt_solver):
    w = opt_solver.w
    n = opt_solver.n
    h = opt_solver.h
    y = opt_solver.y
    s = opt_solver.s
    f = opt_solver.f
    z = opt_solver.z
    qw = opt_solver.qw
    qm = opt_solver.qm
    pf = opt_solver.pf
    pw = opt_solver.pw
    pm = opt_solver.pm

    beta, gamma = opt_solver.unpack_x(x)
    r = (y - f(beta)) * np.sqrt(w)
    v = linalg.VarMat(s**w,
                      (z.T*np.sqrt(w)).T,
                      gamma,
                      n).mat
    inv_v = np.linalg.inv(v)

    obj = 0.5*(r.dot(inv_v.dot(r)) + np.log(np.linalg.det(v)))/h
    obj += 0.5*qw.dot((x - qm)**2)
    obj += 0.5*pw.dot((pf(x) - pm)**2)

    return obj


def true_gradient(x, opt_solver, eps=1e-10):
    xc = x + 0j
    grad = np.zeros(x.size)
    for i in range(x.size):
        xc[i] += eps*1j
        grad[i] = true_objective(xc, opt_solver).imag/eps
        xc[i] -= eps*1j

    return grad


def test_optimization_solver_objective(opt_solver):
    beta = np.random.randn(opt_solver.k_beta)
    gamma = np.random.rand(opt_solver.k_gamma)
    x = np.hstack((beta, gamma))

    obj = true_objective(x, opt_solver)
    my_obj = opt_solver.objective(x)

    assert np.abs(my_obj - obj) < 1e-10


def test_optimization_solver_gradient(opt_solver):
    beta = np.random.randn(opt_solver.k_beta)
    gamma = np.random.rand(opt_solver.k_gamma)
    x = np.hstack((beta, gamma))

    grad = true_gradient(x, opt_solver)
    my_grad = opt_solver.gradient(x)

    assert np.linalg.norm(my_grad - grad) < 1e-10
