# -*- coding: utf-8 -*-
"""
    limetr.core
    ~~~~~~~~~~~

    This module implement the core classes of limetr package.
"""
import numpy as np


class LimeTr:
    def __init__(self,
                 obs_mean,
                 fe_desfun,
                 re_desmat,
                 obs_sd=None,
                 share_obs_sd=True,
                 group_sizes=None,
                 inlier_pct=1.0):
        # data and settings
        self.obs_mean = obs_mean
        self.fe_desfun = fe_desfun
        self.re_desmat = re_desmat
        self.obs_sd = obs_sd
        self.share_obs_sd = share_obs_sd
        self.inlier_pct = inlier_pct

        # dimensions
        self.num_obs = obs_mean.size
        if group_sizes is None:
            self.group_sizes = np.array([1]*self.num_obs)
        else:
            self.group_sizes = group_sizes
        self.num_groups = self.group_sizes.size
        self.dim_beta = self.fe_desfun.shape[1]
        self.dim_gamma = self.re_desmat.shape[1]
        self.dim_fe = self.dim_beta + self.dim_gamma
        self.dim_re = (self.num_groups, self.dim_gamma)
