# -*- coding: utf-8 -*-
"""
    tests.linalg
    ~~~~~~~~~~~

    Test the linalg module.
"""
from limetr import linalg
import numpy as np
import pytest


@pytest.fixture()
def shape():
    return 1, 5


@pytest.fixture()
def mat(shape):
    return np.ones(shape)


@pytest.fixture()
def mat_lm(shape, mat):
    def mv(x):
        return mat.dot(x)

    def trans_mv(y):
        return mat.T.dot(y)

    return linalg.LinearMap(shape, mv, trans_mv)


@pytest.fixture()
def vec_x(shape):
    return np.random.randn(shape[1])


@pytest.fixture()
def vec_y(shape):
    return np.random.randn(shape[0])


@pytest.fixture()
def jac_mat_sf(shape, mat):
    return linalg.SmoothFunction(shape, lambda x: mat.dot(x),
                                 jac_mat=lambda x: mat)


@pytest.fixture()
def jac_fun_sf(shape, mat_lm):
    return linalg.SmoothFunction(shape, mat_lm.dot,
                                 jac_fun=lambda x: mat_lm)


def test_linear_map(mat, mat_lm, vec_x, vec_y):
    assert np.all(mat_lm.dot(vec_x) == mat.dot(vec_x))
    assert np.all(mat_lm.T.dot(vec_y) == mat.T.dot(vec_y))


def test_smooth_fun(jac_mat_sf, jac_fun_sf):
    jac_mat_sf.check()
    jac_fun_sf.check()


@pytest.mark.parametrize(("group_sizes", "dim_gamma"),
                         [(np.array([2, 3, 4]), 2),
                          (np.array([4, 3, 2]), 4)])
@pytest.mark.parametrize("x", [np.random.randn(9), np.random.randn(9, 2)])
def test_varmat_dot(group_sizes, dim_gamma, x):
    num_obs = group_sizes.sum()
    obs_sd = np.random.rand(num_obs) + 0.01
    gamma = np.random.rand(dim_gamma) + 0.01
    re_mat = np.random.randn(num_obs, dim_gamma)

    var_mat = linalg.VarMat(obs_sd, re_mat, gamma, group_sizes)

    assert np.linalg.norm(var_mat.mat.dot(x) - var_mat.dot(x)) < 1e-8


@pytest.mark.parametrize(("group_sizes", "dim_gamma"),
                         [(np.array([2, 3, 4]), 2),
                          (np.array([4, 3, 2]), 4)])
@pytest.mark.parametrize("x", [np.random.randn(9), np.random.randn(9, 2)])
def test_varmat_inv_dot(group_sizes, dim_gamma, x):
    num_obs = group_sizes.sum()
    obs_sd = np.random.rand(num_obs) + 0.01
    gamma = np.random.rand(dim_gamma) + 0.01
    re_mat = np.random.randn(num_obs, dim_gamma)

    var_mat = linalg.VarMat(obs_sd, re_mat, gamma, group_sizes)

    assert np.linalg.norm(var_mat.inv_mat.dot(x) - var_mat.inv_dot(x)) < 1e-8


@pytest.mark.parametrize(("group_sizes", "dim_gamma"),
                         [(np.array([2, 3, 4]), 2),
                          (np.array([4, 3, 2]), 4)])
@pytest.mark.parametrize("x", [np.random.randn(9), np.random.randn(9, 2)])
def test_varmat_diag(group_sizes, dim_gamma, x):
    num_obs = group_sizes.sum()
    obs_sd = np.random.rand(num_obs) + 0.01
    gamma = np.random.rand(dim_gamma) + 0.01
    re_mat = np.random.randn(num_obs, dim_gamma)

    var_mat = linalg.VarMat(obs_sd, re_mat, gamma, group_sizes)

    assert np.linalg.norm(np.diag(var_mat.mat) - var_mat.diag()) < 1e-8


@pytest.mark.parametrize(("group_sizes", "dim_gamma"),
                         [(np.array([2, 3, 4]), 2),
                          (np.array([4, 3, 2]), 4)])
@pytest.mark.parametrize("x", [np.random.randn(9), np.random.randn(9, 2)])
def test_varmat_inv_diag(group_sizes, dim_gamma, x):
    num_obs = group_sizes.sum()
    obs_sd = np.random.rand(num_obs) + 0.01
    gamma = np.random.rand(dim_gamma) + 0.01
    re_mat = np.random.randn(num_obs, dim_gamma)

    var_mat = linalg.VarMat(obs_sd, re_mat, gamma, group_sizes)

    assert np.linalg.norm(np.diag(var_mat.inv_mat) - var_mat.inv_diag()) < 1e-8


@pytest.mark.parametrize(("group_sizes", "dim_gamma"),
                         [(np.array([2, 3, 4]), 2),
                          (np.array([4, 3, 2]), 4)])
@pytest.mark.parametrize("x", [np.random.randn(9), np.random.randn(9, 2)])
def test_varmat_log_det(group_sizes, dim_gamma, x):
    num_obs = group_sizes.sum()
    obs_sd = np.random.rand(num_obs) + 0.01
    gamma = np.random.rand(dim_gamma) + 0.01
    re_mat = np.random.randn(num_obs, dim_gamma)

    var_mat = linalg.VarMat(obs_sd, re_mat, gamma, group_sizes)

    assert np.linalg.norm(
        np.log(np.linalg.det(var_mat.mat)) - var_mat.log_det()) < 1e-8
