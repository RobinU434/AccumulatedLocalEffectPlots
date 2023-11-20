from typing import List, Tuple
import numpy as np
from numpy import ndarray


def get_percentiles(array: ndarray, num_percentiles: int = 10) -> ndarray:
    """get percentiles with evenly such that the data fits in evenly distributed buckets

    Args:
        array (ndarray): data with shape [num_samples]. All samples with the same features you want to have the percentiles of   
        num_percentiles (int, optional): How many percentiles you want to have. Defaults to 10.

    Returns:
        ndarray: percentiles [num_percentiles + 1]
    """
    percentiles = np.linspace(0, 100, num=num_percentiles + 1)
    packed_array = np.percentile(array, percentiles)
    return packed_array


def get_1d_bucket_index(data: ndarray, num_percentiles: int = 10) -> Tuple[ndarray, ndarray]:
    """return for each sample in which percentile it belongs

    Args:
        data (ndarray): data column you want to operate on
        num_percentiles (int, optional): How many percentiles you want to look at. Defaults to 10.

    Returns:
        Tuple[ndarray, ndarray]: indices for each samples in which percentile it belongs [num_samples], Percentile boundaries [num_percentiles + 1]
    """
    bins = get_percentiles(data, num_percentiles)

    # substract -1 because the bins begin at 1
    indices = np.digitize(data, bins) - 1

    # pack maximum element(s) into the last bin
    max_index = np.argwhere(indices == num_percentiles)
    indices[max_index] = num_percentiles - 1 
    return indices, bins


def sort_into_1d_buckets(data: ndarray, indices: ndarray) -> List[ndarray]:
    """sort given data into given buckets

    Args:
        data (ndarray): data you want to split into buckets [num_samples, num_features]
        indices (ndarray): for each sample the index [num_samples]

    Returns:
        List[ndarray]: List of len(unique(indices)) many ndarrays. In each ndarray are the sorted samples. 
    """
    num_indices = np.max(indices) + 1
    result = [[]] * num_indices
    for idx in range(num_indices):
        result[idx] = data[np.where(indices == idx)]
    return result


def sort_into_2d_buckets(X: ndarray, columns: Tuple[int], num_percentiles: int) -> Tuple[List[List[ndarray]], ndarray]:
    """pack data into 2d buckets

    Args:
        X (ndarray): X 
        columns (Tuple[int]): _description_
        num_percentiles (int): _description_

    Returns:
        Tuple[List[List[ndarray]], ndarray]: data put into the 2d buckets
    """
    first_column, second_column = columns

    # sort into first bucket
    first_bucket_index, first_order_percentiles = get_1d_bucket_index(X[:, first_column], num_percentiles)
    packed_data = sort_into_1d_buckets(X, first_bucket_index)

    second_order_percentiles = get_percentiles(X[:, second_column], num_percentiles)

    bins = np.stack([first_order_percentiles, second_order_percentiles])
    
    # widen second_order bins to such that max(data) fits into the last bucket
    widen_second_order_percentiles = second_order_percentiles.copy()
    widen_second_order_percentiles[-1] += 1e-8

    bucket_2d = []
    for bucket_content in packed_data:
        second_order_bucket_index = np.digitize(bucket_content, widen_second_order_percentiles)
        bucket_2d.append(sort_into_1d_buckets(X, second_order_bucket_index))
    
    return bucket_2d, bins
