from statistics import quantiles
from typing import Any, Tuple
import numpy as np
import pandas as pd
import seaborn as sns 
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from src.helper import create_2d_cells, create_buckets2, get_one_dim_buckets, get_1d_quantiles, create_1d_buckets

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
        
        
def ale(
    model,
    X: pd.DataFrame,
    grid_shape: Tuple,
    columns: Tuple,
    show: bool = False,
    std: np.array = np.ones(100), 
    mean: np.array = np.zeros(100)
    ):
    """Accumulative local effect

    Args:
        model (_type_): _description_
        X (np.array): _description_
            We are expecting: shape= (num_data_points, num_columns)
        num_buckets (int): _description_
        columns (np.array): is a tuple that determines which feature column is the one we want to analyze
        show (bool, optional): _description_. Defaults to False.
        std (np.array, optional): value to reset normalization. Defaults to np.ones(100).
        mean (np.array, optional): value to reset normalization. Defaults to np.zeros(100).

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    num_columns = len(columns)
    
    array = X.to_numpy()
    
    if num_columns == 1:
        column_idx = X.columns.get_loc(columns[0])
        buckets, quantiles = create_1d_buckets(X = X, 
                                      column_idx = column_idx,
                                      num_buckets = grid_shape[0])
        # overwrite the initial variable because of num_buckets > len(x)
        num_buckets = len(buckets)
        
        # proceed at first only with just one bucket and one column
        ale_score = np.zeros(num_buckets + 1)
        for idx, bucket in enumerate(buckets):
            y_lower = model.predict(bucket.shift_to_lower(column_idx)).sum()
            y_upper = model.predict(bucket.shift_to_upper(column_idx)).sum()
            
            difference = y_upper - y_lower
            
            ale_score[idx + 1] = difference / len(bucket)
        
        ale_score = np.cumsum(ale_score)
        
        # center the ale_score
        ale_score -= np.trapz(ale_score, x=quantiles) / (quantiles[-1] - quantiles[0])
        
        
        std = std[column_idx]
        mean = mean[column_idx]
        # change quantile distribution
        quantiles = quantiles * std + mean
        
        # plot results
        if show:     
            fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
            # axs[0].plot(np.linspace(array[:, column_idx].min(axis=0), 
            #                     array[:, column_idx].max(axis=0),
            #                     num_buckets + 1), ale_score)
            fig.suptitle(f"First order ALE of feature: '{columns[0]}' \n bins: {grid_shape[0]}")
            axs[0].plot(quantiles, ale_score)
            
            line_length = (ale_score.max() - ale_score.min()) * 0.05 
            line_offset = ale_score.min() - line_length * 1.5
            
            axs[0].eventplot(X[columns[0]] * std + mean, linelengths=line_length, lineoffsets=line_offset)
            
            # axs[0].eventplot(quantiles, linelengths=line_length, lineoffsets=line_offset - line_length * 1.5)
            
            sns.histplot(array[:, column_idx] * std + mean, 
                        ax=axs[1],
                        bins=num_buckets, 
                        kde=False,
                        stat="probability")
            plt.show()
    
    elif num_columns == 2: 
        raise NotImplementedError("Is not implemented yet. I will refer to 2D methods with https://github.com/blent-ai/ALEPython/")
        
    else:
        raise NotImplementedError("More than two columns are not supported")  
    
    return ale_score, quantiles


def sample_ale(
    model_class,
    X: pd.DataFrame,
    y: pd.DataFrame,
    grid_shape: Tuple,
    columns: Tuple,
    shuffle: bool = False,
    num_samples: int = 10,
    std: np.array = np.ones(100), 
    mean: np.array = np.zeros(100), 
    show: bool = False,
    ):
    
    features = X.columns
    
    scores = np.array([])
    accuracies = np.array([])
    
    if show:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    
    for _ in range(num_samples):
        if shuffle:
            sklearn.utils.shuffle(X.to_numpy())
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = model_class()
        model = model.fit(X_train, y_train)
        
        prediction = model.predict(X_test)
        accuracies = np.append(accuracies, accuracy_score(prediction,y_test))
        print(f"Model test accuracy is at {accuracy * 100}%")
        
        score, quantiles = ale(model,
                                pd.DataFrame(X_train, columns=features),
                                columns = columns,
                                grid_shape = grid_shape,
                                show = False,
                                std = std, 
                                mean = mean)
        
        print("score ", score)
        scores = np.append(scores, [[score]])
        
        if show:
            axs[0].plot(quantiles, score, alpha=0.5, color="grey")
            
    if show:
        line_length = (scores.max() - scores.min()) * 0.05 
        line_offset = scores.min() - line_length * 1.5
                
        X_train_df = pd.DataFrame(X_train, columns=features)
        column_idx = X_train_df.columns.get_loc(columns[0])
        
        axs[0].eventplot(X[columns[0]] * std[column_idx] + mean[column_idx], linelengths=line_length, lineoffsets=line_offset)
        
        sns.histplot(X_train_df[columns[0]] * std[column_idx] + mean[column_idx], 
                    ax=axs[1],
                    bins=grid_shape[0], 
                    kde=False,
                    stat="probability")
        
        plt.show()
    
    return scores.reshape(num_samples, len(scores)//num_samples), accuracies , quantiles

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
    data = np.random.normal(size=(100, 2))
    
    dim = 1
    if dim == 1:
        num_buckets = 100
        ale_score = ale(
            model,
            data,
            grid_shape=(num_buckets, ),
            columns=(0,), show=False
            )
    if dim == 2:
        grid_shape = (3,3)
        ale_score = ale(
            model,
            data,
            grid_shape=grid_shape,
            columns=(0,1)
            )