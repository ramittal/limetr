"""
    limetr.optim
    ~~~~~~~~~~~~

    Optimization module for limetr object.
"""
import numpy as np
from scipy.optimize import bisect
import ipopt
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
            self.k_c = self.cf.shape[0]
        else:
            self.k_c = 0
            self.cl = []
            self.cu = []


        self.pf = pf
        self.pm = pm
        self.pw = pw
        self.use_p = (self.pf is not None and
                      self.pm is not None and
                      self.pw is not None)

        if self.use_p:
            self.k_p = self.pf.shape[0]
        else:
            self.k_p = 0
            self.pm = []
            self.pw = []

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
            g += self.pf.jac(x).T.dot(self.pw*(self.pf(x) - self.pm))

        return g

    def initialize_inner_vars(self, x_init=None):
        if x_init is not None:
            assert x_init.size == self.k
            return x_init
        # initialize gamma
        gamma_init = np.repeat(0.1, self.k_gamma)
        # initialize beta
        beta_ones = np.ones(self.k_beta)
        v = self.compute_v(gamma_init)
        jf = self.f.jac(beta_ones)

        mat = jf.T.dot(v.inv_dot(jf))/self.h
        rhs = jf.T.dot(v.inv_dot(self.y))/self.h

        if self.use_q:
            mat += np.diag(self.qw)
            rhs += self.qw*self.qm

        if self.use_p:
            jpf = self.pf.jac(beta_ones)
            mat += (jpf.T*self.pw).dot(jpf)
            rhs += (jpf.T*self.pw).dot(self.pm)

        beta_init = np.linalg.pinv(mat).dot(rhs)

        return np.hstack((beta_init, gamma_init))

    def initialize_outer_vars(self, w_init=None, s_init=None):
        if w_init is not None:
            assert np.abs(w_init.sum() - self.h) < 1e-10
            assert np.all(w_init >= 0.0) and np.all(w_init <= 1.0)
            self.w = w_init
        if self.estimate_s:
            if s_init is not None:
                assert np.all(s_init > 0.0)
                self.s = s_init
            else:
                self.s = np.std(self.y)

    def optimize_inner_vars(self, x, inner_solver_options):
        inner_opt_problem = ipopt.problem(
            n=self.k,
            m=self.k_c,
            problem_obj=self,
            lb=self.bl,
            ub=self.bu,
            cl=self.cl,
            cu=self.cu
        )

        # add options
        for option_name in inner_solver_options.keys():
            inner_opt_problem.addOption(option_name,
                                        inner_solver_options[option_name])

        inner_soln, inner_info = inner_opt_problem.solve(x)

        return inner_soln

    def optimize_outer_vars(self, inner_soln, outer_solver_options):
        # update trimming weights
        beta_soln, gamma_soln = self.unpack_x(inner_soln)
        r = self.compute_r(beta_soln)
        v = self.compute_v(gamma_soln)
        g_w = r*v.inv_dot(r)
        g_w_max = np.linalg.norm(g_w, np.inf)
        if g_w_max != 0:
            g_w /= g_w_max

        self.w = proj_capped_simplex(
            self.w - outer_solver_options["step_size"]*g_w,
            self.h
        )

        # update observation standard deviation
        if self.estimate_s:
            u = self.estimate_u(beta_soln, gamma_soln)
            r = self.y - self.f(beta_soln) -\
                np.sum(self.z*np.repeat(u, self.n, axis=0), axis=1)
            self.s = np.std(r)

    def optimize(self,
                 x_init=None,
                 w_init=None,
                 s_init=None,
                 inner_solver_options=None,
                 outer_solver_options=None):
        x = self.initialize_inner_vars(x_init=x_init)
        self.initialize_outer_vars(w_init=w_init, s_init=s_init)

        iter_count = 0
        err = outer_solver_options["tol"]
        obj = self.objective(x)
        while err >= outer_solver_options["tol"]:
            x = self.optimize_inner_vars(x, inner_solver_options)
            self.optimize_outer_vars(x, outer_solver_options)

            # update err and obj
            obj_new = self.objective(x)
            err = np.abs(obj_new - obj)/np.abs(obj)

            # update iteration information
            iter_count += 1
            if iter_count >= outer_solver_options["max_iter"]:
                print("reach maximum number of iterations.")
                break

        if self.estimate_s:
            return x, self.w, self.s
        else:
            return x, self.w

    def estimate_u(self, beta, gamma):
        r = self.compute_r(beta)
        inv_s2_z = self.z/(self.s**2).reshape(self.N, 1)
        inv_gamma = 1.0/gamma

        r_split = np.split(r, self.idx_split[1:])
        z_split = np.split(self.z, self.idx_split[1:], axis=0)
        inv_s2_z_split = np.split(inv_s2_z, self.idx_split, axis=0)

        u = [np.linalg.solve(inv_s2_z_split[i].T.dot(z_split[i]) +
                             np.diag(inv_gamma),
                             inv_s2_z_split[i].T.dot(r_split[i]))
             for i in range(self.m)]

        return np.vstack(u)

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


def proj_capped_simplex(w, h):
    a = np.min(w) - 1.0
    b = np.max(w) - 0.0

    def f(w_shift):
        return np.sum(np.maximum(np.minimum(w - w_shift, 1.0), 0.0)) - h

    w_shift_soln = bisect(f, a, b)

    return np.maximum(np.minimum(w - w_shift_soln, 1.0), 0.0)
