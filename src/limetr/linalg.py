# -*- coding: utf-8 -*-
"""
    limetr.linalg
    ~~~~~~~~~~~~~

    Linear algebra class for limetr object
"""
import numpy as np
from scipy.linalg import block_diag
import limetr.linalgf as linalgf


class LinearMap:
    """Linear map class, conduct dot product when do not have function form
    only support matrix vector multiplication
    """
    def __init__(self, shape, mv, trans_mv=None):
        self.shape = shape
        self.mv = mv
        self.trans_mv = trans_mv

    def dot(self, x):
        return self.mv(x)

    @property
    def mat(self):
        return np.vstack([self.dot(np.eye(1, np.shape[1], i))
                          for i in range(self.shape[1])]).T

    @property
    def T(self):
        if self.trans_mv is None:
            return None
        else:
            return LinearMap((self.shape[1], self.shape[0]),
                             self.trans_mv,
                             trans_mv=self.mv)


class SmoothFunction:
    """Smooth function class, include function mapping and
    Jacobian matrix or its LinearMap object
    """
    def __init__(self, shape, fun, jac_mat=None, jac_fun=None):
        # check the input
        assert jac_mat is not None or jac_fun is not None
        # pass in the data
        self.shape = shape
        self.fun = fun
        self.jac_mat = jac_mat
        self.jac_fun = jac_fun
        if self.jac_mat is not None:
            self.jac_type = 'jac_mat'
            self.jac = self.jac_mat
        else:
            self.jac_type = 'jac_fun'
            self.jac = self.jac_fun

    def __call__(self, x):
        return self.fun(x)

    def check(self):
        """check the output dimension"""
        x = np.zeros(self.shape[1])
        fun_result = self.fun(x)
        jac_result = self.jac(x)
        assert type(fun_result) == np.ndarray
        assert fun_result.shape == (self.shape[0],)
        if self.jac_type == 'jac_mat':
            assert type(jac_result) == np.ndarray
            assert jac_result.shape == self.shape
        else:
            assert type(jac_result) == LinearMap
            assert jac_result.shape == self.shape


class VarMat:
    def __init__(self, obs_sd, re_mat, gamma, group_sizes):
        # pass in data
        self.obs_sd = obs_sd
        self.re_mat = re_mat
        self.gamma = gamma
        self.group_sizes = group_sizes

        # dimensions
        self.num_groups = len(group_sizes)
        self.num_obs = sum(group_sizes)
        self.dim_gamma = gamma.size

        self.check_input()

        # convert to izmat format
        self.scaled_z = ((self.re_mat*np.sqrt(self.gamma))/
                         self.obs_sd.reshape(self.num_obs, 1))
        # lsvd of izmat
        self.scaled_z_ns = np.minimum(self.group_sizes, self.dim_gamma)
        self.scaled_z_nu = self.group_sizes*self.scaled_z_ns
        self.scaled_z_s = np.zeros(self.scaled_z_ns.sum())
        self.scaled_z_u = np.zeros(self.scaled_z_nu.sum())
        linalgf.izmat.lsvd(self.group_sizes,
                           self.scaled_z_nu,
                           self.scaled_z_ns,
                           self.scaled_z,
                           self.scaled_z_u,
                           self.scaled_z_s)

        # inverse and eig
        self.scaled_inv_z_ns = self.scaled_z_ns
        self.scaled_inv_z_nu = self.scaled_z_nu
        self.scaled_inv_z_u = self.scaled_z_u
        self.scaled_z_s2 = self.scaled_z_s**2
        self.scaled_inv_z_s2 = 1.0/(1.0 + self.scaled_z_s2) - 1.0

        self.scaled_iz_e = linalgf.izmat.izeig(self.num_obs,
                                               self.group_sizes,
                                               self.scaled_z_ns,
                                               self.scaled_z_s2)

    def check_input(self):
        """Ensure the correct input.
        """
        assert self.obs_sd.shape == (self.num_obs,)
        assert self.re_mat.shape == (self.num_obs, self.dim_gamma)

        assert np.all(self.obs_sd > 0.0)
        assert np.all(self.gamma >= 0.0)

    @property
    def mat(self):
        """
        naive implementation of the varmat
        """
        split_idx = np.cumsum(self.group_sizes)[:-1]
        obs_sd_list = np.split(self.obs_sd, split_idx)
        re_mat_list = np.split(self.re_mat, split_idx, axis=0)

        diag_blocks = [
            np.diag(obs_sd_list[i]**2) +
            (re_mat_list[i]*self.gamma).dot(re_mat_list[i].T)
            for i in range(self.num_groups)
        ]

        return block_diag(*diag_blocks)

    @property
    def inv_mat(self):
        """
        naive implementation of the inverse varmat
        """
        split_idx = np.cumsum(self.group_sizes)[:-1]
        obs_sd_list = np.split(self.obs_sd, split_idx)
        re_mat_list = np.split(self.re_mat, split_idx, axis=0)

        diag_blocks = [
            np.linalg.inv(np.diag(obs_sd_list[i] ** 2) +
                          (re_mat_list[i] * self.gamma).dot(re_mat_list[i].T))
            for i in range(self.num_groups)
        ]

        return block_diag(*diag_blocks)

    def dot(self, x):
        """
        dot product with the covariate matrix
        """
        if x.ndim == 1:
            operation = linalgf.izmat.izmv
            scale = self.obs_sd
        elif x.ndim == 2:
            operation = linalgf.izmat.izmm
            scale = self.obs_sd.reshape(self.num_obs, 1)
        else:
            print('unsupported dim of x')
            return None

        return operation(self.scaled_z_nu,
                         self.scaled_z_ns,
                         self.group_sizes,
                         self.scaled_z_u,
                         self.scaled_z_s2,
                         x*scale)*scale

    def inv_dot(self, x):
        """
        inverse dot product with the covariate matrix
        """
        if x.ndim == 1:
            operation = linalgf.izmat.izmv
            scale = self.obs_sd
        elif x.ndim == 2:
            operation = linalgf.izmat.izmm
            scale = self.obs_sd.reshape(self.num_obs, 1)
        else:
            print('unsupported dim of x')
            return None

        return operation(self.scaled_inv_z_nu,
                         self.scaled_inv_z_ns,
                         self.group_sizes,
                         self.scaled_inv_z_u,
                         self.scaled_inv_z_s2,
                         x/scale)/scale

    def diag(self):
        """
        return the diagonal of the matrix
        """
        return linalgf.izmat.izdiag(self.num_obs,
                                    self.scaled_z_nu,
                                    self.scaled_z_ns,
                                    self.group_sizes,
                                    self.scaled_z_u,
                                    self.scaled_z_s2)*self.obs_sd**2

    def inv_diag(self):
        """
        return the diagonal of the inverse covariate matrix
        """
        return linalgf.izmat.izdiag(self.num_obs,
                                    self.scaled_inv_z_nu,
                                    self.scaled_inv_z_ns,
                                    self.group_sizes,
                                    self.scaled_inv_z_u,
                                    self.scaled_inv_z_s2)/self.obs_sd**2

    def log_det(self):
        """
        returns the log determinant of the covariate matrix
        """
        return (np.sum(np.log(self.scaled_iz_e)) +
                2.0*np.sum(np.log(self.obs_sd)))
