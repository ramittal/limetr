# -*- coding: utf-8 -*-
"""
    tests.core
    ~~~~~~~~~~

    Test the core module.
"""
import numpy as np
import pytest
from limetr import core
from limetr import utils


# settings for the test problem
true_group_sizes = np.array([10, 5, 5])
dim_beta = 2
dim_gamma = 2

num_obs = true_group_sizes.sum()
num_groups = true_group_sizes.size
fe_mat = np.random.randn(num_obs, dim_beta)
fe_fun = utils.create_linear_smooth_fun(fe_mat)
re_mat = fe_mat.copy()
true_beta = np.random.randn(dim_beta)
true_gamma = np.random.rand(dim_gamma) + 0.01
true_obs_sd = np.random.rand(num_obs) + 0.01
true_re = np.repeat(np.random.randn(num_groups, dim_gamma),
                    true_group_sizes,
                    axis=0)
true_obs_err = np.random.randn(num_obs)*true_obs_sd
obs_mean = fe_fun(true_beta) + np.sum(re_mat*true_re, axis=1) + true_obs_err


@pytest.mark.parametrize("name", ["beta", None])
@pytest.mark.parametrize("size", [2, 5])
def test_core_variable(name, size):
    variable = core.LimeTrVariable(size, name=name)
    assert variable.values is None
    assert variable.priors is None

    values = np.random.randn(size)
    variable.set_values(values)
    assert np.all(variable.values == values)


@pytest.mark.parametrize("obs_sd", [None, true_obs_sd])
@pytest.mark.parametrize("share_obs_sd", [True, False])
@pytest.mark.parametrize("group_sizes", [None, true_group_sizes])
@pytest.mark.parametrize("inlier_pct", [1.0, 0.95])
def test_core_init(obs_sd, share_obs_sd, group_sizes, inlier_pct):
    lt = core.LimeTr(obs_mean,
                     fe_fun,
                     re_mat,
                     obs_sd=obs_sd,
                     share_obs_sd=share_obs_sd,
                     group_sizes=group_sizes,
                     inlier_pct=inlier_pct)

    if group_sizes is None:
        assert np.all(lt.group_sizes == 1.0)
        assert lt.group_sizes.size == lt.num_obs
    else:
        assert np.all(group_sizes == group_sizes)

    if obs_sd is None:
        if share_obs_sd:
            assert lt.variables["delta"].size == 1
        else:
            assert lt.variables["delta"].size == lt.num_groups
    else:
        assert "delta" not in lt.variables.keys()
        assert np.all(lt.obs_sd == obs_sd)

    assert lt.variables["beta"].priors is None
    assert len(lt.variables["gamma"].priors) == 1
    if obs_sd is None:
        assert len(lt.variables["delta"].priors) == 1


@pytest.fixture()
def lt():
    return core.LimeTr(obs_mean,
                       fe_fun,
                       re_mat,
                       obs_sd=true_obs_sd,
                       share_obs_sd=False,
                       group_sizes=true_group_sizes,
                       inlier_pct=1.0)



