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
