import imp
from random import random
from typing import Any
import numpy as np
import pandas as pd 
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from src.helper import create_buckets

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

        
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
            We are expecting: shape= (num_data_points, num_columns)
        num_buckets (int): _description_
        columns (np.array): is a tuple that determines which feature column is the one we want to analyze
    """
    
    num_columns = len(columns)
    
    if num_columns > 2:
        raise NotImplementedError("More than two columns are not supported")
    
    buckets = create_buckets(X=X, 
                             columns=columns,
                             num_buckets=num_buckets)
    
    # proceed at first only with just one bucket and one column
    ale_score = np.zeros(num_buckets + 1)
    for idx, bucket in enumerate(buckets):
        column = columns[0]
        y_lower = model.predict(bucket.shift_to_lower(column)).sum()
        y_upper = model.predict(bucket.shift_to_upper(column)).sum()
        
        difference = y_upper - y_lower
        
        ale_score[idx + 1] = difference / len(bucket)
    
    ale_score = np.cumsum(ale_score)
    
    # center the ale_score
    ale_score -= ale_score.mean()
    
    
    
    print(X[0:2, :]])
    
    
    # the random data
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    # plot ale score 
    axAle = plt.axes(rect_scatter)
    # axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    # axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    # axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    print(columns)
    x_lim = (X[:, columns].min(axis=0), X[:, columns].max(axis=0))
    print(x_lim)
    pltAle = axAle.plot(np.linspace(x_lim[0],
                                  x_lim[1],
                                  num_buckets + 1), 
                      ale_score)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    axAle.set_xlim(x_lim)
    axAle.set_ylim((ale_score.min(), ale_score.max()))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(X[:, column], bins=num_buckets)  # horizontal histogram 
    # axHisty.hist(y, bins=bins, orientation='horizontal')  # vertical histogram

    axHistx.set_xlim(axAle.get_xlim())
    # axHisty.set_ylim(axScatter.get_ylim())

    plt.show()
    
    # plt.plot(np.linspace(X[:, columns].min(axis=0), 
    #                      X[:, columns].max(axis=0),
    #                      num_buckets + 1), ale_score)
    # # plt.axis("scaled")
    # plt.scatter(X.T[0], X.T[1])
    # plt.show()
    
    return ale_score

class Model:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return x.sum(axis=1)
    
    def forward(self, x):
        return self.__call__(x)
    
    def predict(self, x):
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
    data = np.random.random((10, 2))
    num_buckets = 3
    ale_score = ale(
        model,
        data,
        num_buckets=num_buckets,
        columns=(0,)
        )
    
    plt.plot(np.linspace(0,1, num_buckets + 1), ale_score)
    plt.axis("scaled")
    plt.scatter(data.T[0], data.T[1])
    plt.show()
    