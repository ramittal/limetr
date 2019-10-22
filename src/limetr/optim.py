"""
    limetr.optim
    ~~~~~~~~~~~~

    Optimization module for limetr object.
"""
import numpy as np
from . import linalg
from . import utils


class OptimizationSolver:
    def __init__(self, n, y, f, z, s=None,
                 bl=None, bu=None,
                 qm=None, qw=None,
                 cf=None, cl=None, cu=None,
                 pf=None, pm=None, pw=None,
                 p=1.0):
        # pass in data
        self.n = n
        self.y = y
        self.f = f
        self.z = z
        self.s = s
        self.estimate_s = self.s is None
        if self.estimate_s:
            self.s = np.repeat(np.std(self.y), self.y.size)

        self.bl = bl
        self.bu = bu
        self.use_b = (self.bl is not None and
                      self.bu is not None)

        self.qm = qm
        self.qw = qw
        self.use_q = (self.qm is not None and
                      self.qw is not None)

        self.cf = cf
        self.cl = cl
        self.cu = cu
        self.use_c = (self.cf is not None and
                      self.cl is not None and
                      self.cu is not None)

        if self.use_c:
            self.constraints = self.cf
            self.jacobian = self.cf.jac

        self.pf = pf
        self.pm = pm
        self.pw = pw
        self.use_p = (self.pf is not None and
                      self.pm is not None and
                      self.pw is not None)

        # extract the dimension
        self.k_beta = self.f.shape[1]
        self.k_gamma = self.z.shape[1]
        self.k = self.k_beta + self.k_gamma
        self.idx_beta = slice(0, self.k_beta)
        self.idx_gamma = slice(self.k_gamma, self.k)
        self.idx_split = np.cumsum(np.insert(self.n, 0, 0))[:-1]

        self.m = len(self.n)
        self.N = sum(self.n)

        # trimming settings
        self.h = int(p*self.N)
        self.p = self.h/self.N
        self.w = np.repeat(self.p, self.N)

    def objective(self, x):
        beta, gamma = self.unpack_x(x)
        r = self.compute_r(beta)
        v = self.compute_v(gamma)

        val = 0.5*(r.dot(v.inv_dot(r)) + v.log_det())/self.h

        # add priors
        if self.use_q:
            val += 0.5*self.qw.dot((x - self.qm)**2)

        if self.use_p:
            val += 0.5*self.pw.dot((self.pf(x) - self.pm)**2)

        return val

    def gradient(self, x):
        beta, gamma = self.unpack_x(x)
        r = self.compute_r(beta)
        v = self.compute_v(gamma)
        inv_v_r = v.inv_dot(r)
        inv_v_z = v.inv_dot(self.z)

        # gradient for beta
        g_beta = -self.f.jac(beta).T.dot(inv_v_r)/self.h

        # gradient for gamma
        g_gamma = 0.5*(np.sum(self.z*inv_v_z, axis=0) -
                       np.sum(np.add.reduceat(inv_v_z.T*r,
                                              self.idx_split,
                                              axis=1)**2, axis=1))/self.h

        g = np.hstack((g_beta, g_gamma))
        # gradient from prior
        if self.use_q:
            g += self.qw*(x - self.qm)

        if self.use_p:
            g += self.pf.jac.T.dot(self.pw*(self.pf(x) - self.pm))

        return g

    def unpack_x(self, x):
        beta = x[self.idx_beta]
        gamma = x[self.idx_gamma]
        # safe guard for the variance
        gamma[gamma <= 0.0] = 0.0
        return beta, gamma

    def compute_r(self, beta):
        return (self.y - self.f(beta))*self.w

    def compute_v(self, gamma):
        return linalg.VarMat(self.s**self.w,
                             self.z*self.w.reshape(self.N, 1),
                             gamma,
                             self.n)
