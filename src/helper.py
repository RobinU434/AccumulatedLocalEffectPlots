import unittest

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple
from torch import grid_sampler

from src.bucket import Bucket, Cell


def create_buckets2(X: np.array, 
                  columns: int,
                  grid_shape: Tuple):
    
    # if there is only one column there only can be a single value in gridshape
    assert len(grid_shape) == 1
    
    num_buckets = int(grid_shape[0])
    column = columns[0]
    
    return get_one_dim_buckets(X, num_buckets=num_buckets, column=column)
        
    # elif num_columns == 2:
    #     num_buckets_horizontal = int(grid_shape[0])
    #     num_buckets_vertical = int(grid_shape[1])
    #     
    #     horizontal_column, vertical_column = grid_shape
    #     
    #     # fill the horizontal buckets 
    #     horizontal_buckets = get_one_dim_buckets(X, num_buckets=num_buckets_horizontal, column=horizontal_column)
    #     # ^x2
    #     # |   b1  b2  b3  b4
    #     # |   |   |   |   |
    #     # |   |   |   |   |
    #     # |   |   |   |   |
    #     # ----------------->  x1
    #     # shown are the upper borders from bucket 1 (b1) to bucket 4 (b4)
    #     
    #     # insert the vertical separation
        
            
def get_one_dim_buckets(X, num_buckets, column):
    # in this version the limits will be determined by the individual data distribution 
    # copy data to a working frame
    X_copy = np.copy(X)
    # sort data
    X_copy = X_copy[X_copy[:, column].argsort()]
    # compute step size aka number of data points per bucket
    num_data_points = len(X)
    step_size = num_data_points / num_buckets
    
    if step_size < 1:
        step_size = 1
        num_buckets
        raise Warning("You can't allocate more buckets than data points: num_buckets will be set to len(X) and step size will be set to one")
    
    
    buckets = [Bucket() for _ in range(num_buckets)]
    last_max = None
    for idx, bucket in enumerate(buckets):
        data_points = X_copy[round(idx * step_size): round((idx + 1) * step_size)]
        
        # to have no spaces between buckets    
        if idx == 0:
            lower_limit = data_points.min(axis=0)[column]
        else:
            lower_limit = last_max
        
        upper_limit = data_points.max(axis=0)[column]
        last_max = upper_limit
        
        # fill values into object
        bucket.container = data_points
        bucket.lower_limit = lower_limit
        bucket.upper_limit = upper_limit
                        
    return buckets


def create_1d_buckets(X: pd.DataFrame, 
                  column_idx: int,
                  num_buckets: int):
        
    # get quantiles to split the DataFrame X
    quantiles, num_buckets = get_1d_quantiles(X, column_idx, num_buckets)
    
    buckets = [Bucket() for _ in range(num_buckets)]
    for bucket_idx, bucket in enumerate(buckets):
        bucket.lower_limit = quantiles[bucket_idx]
        bucket.upper_limit = quantiles[bucket_idx + 1]
        
        data = X[(X.iloc[:, column_idx] >= bucket.lower_limit) &
                    (X.iloc[:, column_idx] <= bucket.upper_limit)]
        
        # fill bucket
        if type(data) == pd.DataFrame:
            data = data.to_numpy()    
        bucket.container = data
        
    return buckets, quantiles
    
    
def create_2d_cells(X: pd.DataFrame, 
                  columns: Tuple,
                  grid_shape: Tuple):
    
    num_columns = len(columns)
    
    assert num_columns == len(grid_shape)
    
    q1, num_row_buckets = get_1d_quantiles(X, columns[0], grid_shape[0])
    q2, num_col_buckets = get_1d_quantiles(X, columns[1], grid_shape[1])
    
    # fill the buckets in 2d
    cells = [[Cell() for _ in range(grid_shape[0])] for _ in range(grid_shape[1])]
    for row_idx, cell_row in enumerate(cells):
        for col_idx, cell in enumerate(cell_row):
            cell.lower_limit = np.array([q1[row_idx], q2[col_idx]], dtype=np.float64)
            cell.upper_limit = np.array([q1[row_idx + 1], q2[col_idx + 1]], dtype=np.float64)
                        
            cell.container = X[(X[:, columns[0]] >= cell.lower_limit[0]) & 
                                (X[:, columns[0]] <= cell.upper_limit[0]) &
                                (X[:, columns[1]] >= cell.lower_limit[1]) & 
                                (X[:, columns[1]] <= cell.upper_limit[1])]
    return cells

        
def get_1d_quantiles(X: pd.DataFrame,
                     column_idx: int,
                     num_buckets: int):
    """return quantiles from 0% to 100% coverage including min and max 

    Args:
        X (pd.DataFrame): _description_
        column_idx (int): _description_
        num_buckets (int): _description_

    Returns:
        _type_: _description_
    """
    
    if type(column_idx) == str:
        column_idx = X.columns.get_loc(column_idx)
        
    # convert X to numpy array
    if type(X) == pd.DataFrame:
        X = X.to_numpy()
    
    # increase the num_buckets to also take min and max into account
    num_buckets = num_buckets + 1
    
    quantiles = np.unique(
        np.quantile(
            X[:, column_idx], np.linspace(0, 1, num_buckets), method="lower"
        )
    )
    num_buckets = len(quantiles) - 1
    
    return quantiles, num_buckets
    

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
            
    def test_1d_quantiles(self):
        X = np.arange(24).reshape(6,4)
        num_buckets = 2
        column = 0
        quantiles, bins = get_1d_quantiles(X, column, num_buckets)
        self.assertEqual(quantiles.tolist(), [0, 8, 20])
        
if __name__ == '__main__':
    unittest.main()