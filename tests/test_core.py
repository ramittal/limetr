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
            assert lt.dim_delta == 1
        else:
            assert lt.dim_delta == lt.num_groups
    else:
        assert lt.dim_delta == 0
        assert np.all(lt.obs_sd == obs_sd)

    assert len(lt.beta_priors) == 0
    assert len(lt.gamma_priors) == 1
    if lt.dim_delta == 0:
        assert len(lt.delta_priors) == 0
    else:
        assert len(lt.delta_priors) == 1
