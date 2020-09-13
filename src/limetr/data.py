"""
    data
    ~~~~

    Data module.
"""
from typing import List, Union
from dataclasses import dataclass, field
import numpy as np
from limetr.utils import check_size, has_no_repeat


@dataclass
class Data:
    obs: Union[np.ndarray, List[float]] = field(repr=False)
    obs_se: Union[np.ndarray, List[float]] = field(default=None, repr=False)
    group_sizes: Union[np.ndarray, List[int]] = field(default=None, repr=False)
    index: Union[np.ndarray, List[int]] = field(default=None, repr=False)

    def __post_init__(self):
        self.obs = np.asarray(self.obs)
        self.num_obs = self.obs.size

        self.obs_se = np.ones(self.num_obs) if self.obs_se is None else np.asarray(self.obs_se)
        self.group_sizes = np.array([1]*self.num_obs) if self.group_sizes is None else np.asarray(self.group_sizes)
        self.index = np.arange(self.num_obs) if self.index is None else np.asarray(self.index)

        self.num_groups = self.group_sizes.size

    def check_attr(self):
        check_size(self.obs, self.num_obs, attr_name='obs')
        check_size(self.obs_se, self.num_obs, attr_name='obs_se')
        check_size(self.group_sizes, self.num_groups, attr_name='group_sizes')
        check_size(self.index, self.num_obs, attr_name='index')

        assert all(self.obs_se > 0.0), "Numbers in obs_se must be positive."
        assert all(self.group_sizes > 0.0), "Numbers in group_sizes must be positive."
        assert np.issubdtype(self.group_sizes.dtype, int), "Numbers in group_sizes must be integer."
        assert has_no_repeat(self.index), "Numbers in index must be unique."

    def __repr__(self):
        return f"Data(num_obs={self.num_obs}, num_groups={self.num_groups})"
