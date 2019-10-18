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


@pytest.mark.parametrize(("dtype", "dparams"),
                         [("gaussian", np.array([[0.0]*size, [1.0]*size])),
                          ("uniform", np.array([[0.0]*size, [1.0]*size]))])
@pytest.mark.parametrize(("ptype", "fun"),
                         [("direct_prior", None),
                          ("function_prior", smooth_fun)])
@pytest.mark.parametrize("name", [None, "", "test prior"])
def test_prior(ptype, dtype, dparams, fun, name):
    prior = stats.Prior(ptype, dtype, dparams, fun=fun, name=name)
    prior.check()
