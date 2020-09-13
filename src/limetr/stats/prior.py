"""
    prior
    ~~~~~

    Prior module.
"""
from typing import Union
from dataclasses import dataclass, field
from limetr.linalg import SmoothMapping
from .distribution import Distribution


@dataclass
class Prior:
    distr: Distribution = field(repr=False)
    mapping: Union[SmoothMapping, None] = field(default=None, repr=False)
    name: str = field(default=None, repr=False)

    def __post_init__(self):
        self.shape = (self.distr.size,) if self.mapping is None else self.mapping.shape
        self.distr_name = type(self.distr).__name__
        if self.distr_name == 'Laplace' and (not self.is_direct()):
            raise ValueError("Do not support functional Laplace prior.")
        if self.name is None:
            prior_type = 'Direct' if self.is_direct() else 'Functional'
            self.name = ' '.join([prior_type, self.distr_name])

    def is_direct(self) -> bool:
        return self.mapping is None

    def __repr__(self):
        return f"Prior(shape={self.shape}, distribution={self.distr_name})"
