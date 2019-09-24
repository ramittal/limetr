# -*- coding: utf-8 -*-
"""
    tests.utils
    ~~~~~~~~~~~

    Test the utils module.
"""
from limetr import utils
import numpy as np
import pytest


@pytest.fixture()
def shape():
    return 2, 5


@pytest.fixture()
def size():
    return 5


@pytest.fixture()
def vec(shape):
    return np.random.randn(shape[1])


@pytest.fixture()
def mat(shape):
    return np.random.randn(*shape)


def test_create_linear_smooth_fun(mat, vec):
    sf = utils.create_linear_smooth_fun(mat)
    sf.check()
    assert sf.jac_type == "jac_mat"
    assert np.all(sf.fun(vec) == mat.dot(vec))
    assert np.all(sf.jac(vec) == mat)


def test_create_dummy_uniform_distr(size):
    udistr = utils.create_dummy_uniform_distr(size)
    assert np.all(udistr.lb == -np.inf)
    assert np.all(udistr.ub == np.inf)


def test_create_dummy_gaussian_distr(size):
    gdistr = utils.create_dummy_gaussian_distr(size)
    assert np.all(gdistr.mean == 0.0)
    assert np.all(gdistr.sd == np.inf)


def test_create_positive_uniform_distr(size):
    udistr = utils.create_positive_uniform_distr(size)
    assert np.all(udistr.lb == 0.0)
    assert np.all(udistr.ub == np.inf)


def test_create_negative_uniform_distr(size):
    udistr = utils.create_negative_uniform_distr(size)
    assert np.all(udistr.lb == -np.inf)
    assert np.all(udistr.ub == 0.0)
