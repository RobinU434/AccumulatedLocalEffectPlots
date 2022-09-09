from ast import Num
from typing import Tuple
import numpy as np
import pandas as pd

from src.bucket import Bucket

def create_buckets(X: np.array, 
                  columns: Tuple,
                  num_buckets):
    num_columns = len(columns)
    
    mins = X[:, columns].min(axis=0)
    maxs = X[:, columns].max(axis=0)
    
    if num_columns == 1: 
        column = columns[0]
        base_limits = np.linspace(float(mins), float(maxs), num_buckets + 1)
        # edit lowest border to include those point  which are at the lowest point
        epsilon = 1e-10
        base_limits[0] -= epsilon
        limits = np.array([base_limits[:-1], base_limits[1:]])  # first lower limit then upper limit
        
        buckets = [ Bucket(lower_limit=ll, upper_limit=up) for ll, up in limits.T]
        
        # caution: not runtime efficient
        # better: sort X wrt. the feature index and and then these partitioned lists
        for x in X:
            for bucket in buckets:
                bucket.conditional_add(x, x[column])
                
        return buckets
                    
            
    # span buckets
    buckets_borders = np.zeros((len(columns), num_buckets))
    for i in range(num_columns):
        buckets_borders[i] = np.linspace(mins[i], maxs[i], num_buckets + 1)[1:]
    # buckets_borders = np.array(buckets_borders).T  # transform to numpy array and format: np.array([border1_columnN, border2_columnN], [border1_columnN', border2_columnN']])
    
    if num_columns == 2:
        raise NotImplementedError
        
    # fill buckets
    # first into buckets along the first axis from columns 
    df = pd.DataFrame(np.copy(X).T)
    data = pd.DataFrame(np.copy(X))
    buckets = []
    already_added_index = set()
    for limit in buckets_borders[0]:
        print(limit)
        temp_data = df[df[columns[0]] <= limit]
        index_to_add = set(temp_data.index) - already_added_index
        buckets.append(np.array(data[index_to_add]))
        already_added_index = already_added_index.union(index_to_add)
