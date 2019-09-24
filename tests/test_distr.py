# -*- coding: utf-8 -*-
"""
    tests.distr
    ~~~~~~~~~~~

    Test the distr module.
"""
from limetr import distr
import numpy as np


def test_gaussian_distr():
    size = 5
    mean = np.zeros(size)
    sd = np.ones(size)
    g_distr = distr.Gaussian(mean, sd)
    g_distr.check()

def test_uniform_distr():
    size = 5
    lb = np.zeros(size)
    ub = np.ones(size)
    u_distr = distr.Uniform(lb, ub)
    u_distr.check()
