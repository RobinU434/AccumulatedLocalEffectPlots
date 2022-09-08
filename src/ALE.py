import imp
from random import random
from typing import Any
import numpy as np
import pandas as pd 
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from helper import popcols, poprows

import matplotlib.pyplot as plt

        
def ale(
    model,
    X: np.array,
    num_buckets: int,
    columns: np.array
    ):
    """_summary_

    Args:
        model (_type_): _description_
        X (np.array): _description_
            We are expecting: np.array([column1:np.array, column2.np.array, ...])   -> shape= (num_columns, num_data_points)
        num_buckets (int): _description_
        columns (np.array): is a tuple that determines which feature column is the one we want to analyze
    """
    
    num_columns = len(columns)
    
    if num_columns > 2:
        raise NotImplementedError("More than two columns are not supported")
    
    # span buckets
    mins = X[columns, :].min(axis=1)
    maxs = X[columns, :].max(axis=1)
    
    if type(mins) == float:
        mins  = np.array([mins])
    
    if type(maxs) == float:
        maxs  = np.array([maxs])
             
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
    
    

class Model:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return x.sum()
    
    def forward(self, x):
        return self.__call__(x)
    


def gen_data(plot: bool= False):
    """
    recrate data example from "interpretable machine learning" book chapter 8.2.4
    """
    # random_seed
    np.random.seed(123456789)
    
    # span the mesh
    num_points = 1000
    x1 = np.linspace(0,1, num_points)
    x2 = np.linspace(0,1, num_points)
    x1x, x2x = np.meshgrid(x1, x2)
    
    # underlying function
    zz = x1x + x2x
    # create rectangular exception at the bottom left
    zz[(x1x >= 0.7) & (x2x <= 0.3)] = 2
    
    # scatter data points along linear function
    # operate on indices 
    y1 = np.sort(np.random.uniform(0, num_points, 40))
    y2 = np.copy(y1) + np.random.normal(0, 100, 40).astype(np.int32)
    # trim y2
    y2[y2 < 0] = 0
    y2[y2 >= num_points] = num_points - 1

    
    if plot:
        plt.contourf(x1, x2, zz)
        plt.scatter(y1 / num_points, y2 / num_points, color="k")
        plt.axis("scaled")
        plt.colorbar()
        plt.show()
    
    return x1x, x2x, zz, y1, y2
    
    
if __name__ == "__main__":
    # gen_data(plot=True)
    model = Model()
    np.random.seed(123456789)
    data = np.random.random((5, 10))
    
    print(data)
    ale(
        model,
        data,
        num_buckets=3,
        columns=(0,)
        )
    