from typing import List, Tuple
import numpy as np
from numpy import ndarray


def get_percentiles(array: ndarray, num_percentiles: int = 10) -> ndarray:
    percentiles = np.linspace(0, 100, num=num_percentiles + 1)
    packed_array = np.percentile(array, percentiles)
    return packed_array


def get_1d_bucket_index(data: ndarray, num_percentiles: int = 10) -> Tuple[ndarray, ndarray]:
    bins = get_percentiles(data, num_percentiles)
    # make last boundary just a bit bigger such that max(data) fits into the last bucket
    widen_bins = bins.copy()
    widen_bins[-1] += 1e-8
    
    indices = np.digitize(data, widen_bins)
    return indices, bins


def sort_into_buckets(data: ndarray, indices: ndarray) -> List[ndarray]:
    num_indices = len(np.unique(indices))
    result = [[]] * num_indices
    for idx in range(num_indices):
        result[idx] = data[np.where(indices == (idx + 1))]
    return result
