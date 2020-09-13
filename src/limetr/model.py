"""
    model
    ~~~~~

    Model module, including fixed and random effects model.
"""
from typing import List, Union, Callable
from warnings import warn
import numpy as np
from limetr.linalg import SmoothMapping, LinearMapping
from limetr.stats import Prior, Uniform


def filter_priors(priors: List[Prior], condition: Callable) -> Union[Prior, None]:
    filtered_priors = list(filter(condition, priors))
    if len(filtered_priors) > 1:
        raise ValueError("There are at least two priors with same type, please combine the prior first.")
    prior = None if len(filtered_priors) == 0 else filtered_priors[0]
    return prior


def check_prior_size(prior: Union[Prior, None], size: int):
    if prior is not None:
        assert prior.shape[-1] == size, f"{prior.name} size not match, variable should have size={size}, " \
                                        f"but prior has shape={prior.shape}."


class Model:
    def __init__(self, mapping: SmoothMapping, priors: List[Prior]):
        self.mapping = mapping
        self.shape = self.mapping.shape
        self.num_obs = self.shape[0]
        self.num_var = self.shape[1]

        # direct priors
        self.prior_gaussian = filter_priors(priors, lambda p: p.is_direct() and (p.distr_name == 'Gaussian'))
        self.prior_uniform = filter_priors(priors, lambda p: p.is_direct() and (p.distr_name == 'Uniform'))
        self.prior_laplace = filter_priors(priors, lambda p: p.is_direct() and (p.distr_name == 'Laplace'))

        # functional priors
        self.fun_prior_gaussian = filter_priors(priors, lambda p: not p.is_direct() and (p.distr_name == 'Gaussian'))
        self.fun_prior_uniform = filter_priors(priors, lambda p: not p.is_direct() and (p.distr_name == 'Uniform'))

        self.priors = [self.prior_gaussian, self.prior_uniform, self.prior_laplace,
                       self.fun_prior_gaussian, self.fun_prior_uniform]

        self.check_attr()

    def check_attr(self):
        for prior in self.priors:
            check_prior_size(prior, self.num_var)


class FixedEffectsModel(Model):
    def __init__(self, mapping: SmoothMapping, priors: List[Prior]):
        super().__init__(mapping, priors)


class RandomEffectsModel(Model):
    def __init__(self, mapping: SmoothMapping, priors: List[Prior]):
        super().__init__(mapping, priors)
        if not isinstance(self.mapping, LinearMapping):
            raise ValueError("Random effects model mapping must be linear.")

        if self.prior_uniform is None:
            self.prior_uniform = Prior(distr=Uniform(lb=0.0, ub=np.inf, size=self.num_var))
        else:
            if any(self.prior_uniform.distr.lb < 0.0):
                warn("Random effects variance has to be positive, adjust the uniform prior.")
                self.prior_uniform.distr.lb[self.prior_uniform.distr.lb < 0.0] = 0.0
                self.prior_uniform.distr.ub[self.prior_uniform.distr.ub < 0.0] = 0.0
