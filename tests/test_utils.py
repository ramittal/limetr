# -*- coding: utf-8 -*-
"""
    tests.utils
    ~~~~~~~~~~~

    Test the utils module.
"""
import limetr
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
    fun = utils.create_linear_smooth_fun(mat)
    fun.check()
    assert fun.jac_type == "jac_mat"
    assert np.all(fun(vec) == mat.dot(vec))
    assert np.all(fun.jac(vec) == mat)


@pytest.mark.parametrize("size", [5, 0])
def test_create_positive_uniform_dparams(size):
    dparams = utils.create_positive_uniform_dparams(size)
    assert dparams.shape == (2, size)
    assert np.all(dparams[0] == 0.0)
    assert np.all(np.isposinf(dparams[1]))


@pytest.mark.parametrize("size", [5, 0])
def test_create_negative_uniform_dparams(size):
    dparams = utils.create_negative_uniform_dparams(size)
    assert dparams.shape == (2, size)
    assert np.all(np.isneginf(dparams[0]))
    assert np.all(dparams[1] == 0.0)
