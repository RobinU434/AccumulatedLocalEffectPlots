from argparse import ArgumentError
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd

class Bucket:
    """
    Is a class to represent buckets. 
    A bucket is an object that contains other objects that fall between the lower and the upper limit.
    
    This class is particularly build to host a bucket with identically shaped numpy arrays
    """
    def __init__(self, lower_limit=0, upper_limit=1, columns: list = []) -> None:
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self._container = [] # list of numpy arrays: column -> feature, row -> data point
        
    @property 
    def container(self):
        return self._container
    
    @container.setter    
    def container(self, item):
        self._container = list(item)
        
    def add(self, element):
        self._container.append(element)
    
    def conditional_add(self, element, feature):
        if self.is_inside(feature):
            self.add(element)
    
    def is_inside(self, feature):
        """determines if the requested feature falls into the limits

        Args:
            feature (float) 

        Returns:
            bool: if feature is inside limits 
        """
        
        if feature > self.lower_limit and feature <= self.upper_limit:
            return True
        return False
    
    def shift_to_lower(self, feature_idx):
        """returns self._container with the requested feature idx shifted to the lower_limit

        Args:
            feature_idx (int): idx that hast to be inside an element of an array inside the container

        Returns:
            numpy.array: container as numpy array
        """
        
        res = np.copy(np.array(self._container))
        res[:, feature_idx] = self.lower_limit
        
        return res 
    
    
    def shift_to_upper(self, feature_idx):
        """returns self._container with the requested feature idx shifted to the upper

        Args:
            feature_idx (int): idx that hast to be inside an element of an array inside the container

        Returns:
            numpy.array: container as numpy array
        """
        
        res = np.copy(np.array(self._container))
        res[:, feature_idx] = self.upper_limit
        
        return res 
        
    def __repr__(self) -> str:
        data = pd.DataFrame(self.container)
        return f"lower_limit: {self.lower_limit} \n upper_limit: {self.upper_limit} \n container: {data.head()} \n num_elements: {len(self)} \n"
    
    def __len__(self) -> int:
        return len(self._container)
        
        
class Cell:
    def __init__(self, lower_limit: List = np.array([]), upper_limit: List = np.array([]), dimensions: int = 2):
        self.lower_limit: np.array = lower_limit
        self.upper_limit: np.array = upper_limit
        
        # test if the limits have the same dimensionality
        if self.lower_limit.shape != self.upper_limit.shape:
            raise ValueError("Dimensions of lower_limit and upper limit hast to be the same.")
        
        self.dimension = dimensions
        
        self._container = []
        
    @property 
    def container(self):
        return self._container
    
    @container.setter    
    def container(self, item):
        self._container = list(item)
        
    def add(self, element):
        self._container.append(element)
    
    def conditional_add(self, element, feature):
        if self.is_inside(feature):
            self.add(element)
            
    def is_inside(self, features: Tuple):
        """
        determines if the requested feature falls into the limits

        Args:
            features (Tuple): tuple with the features which I would like to check if they're inside of the limits

        Returns:
            bool: if feature is inside limits 
        """
        
        raise NotImplementedError
        
            
    def shift_to_lower(self, dimension: int, feature_idx: int):
        """_summary_

        Args:
            dimension (int): index from lower limit to state which value to inset into data
            feature_idx (int): index for data to determine in which column to insert the lower limit

        Raises:
            ArgumentError: _description_

        Returns:
            _type_: _description_
        """
        if dimension >= self.dimension:
            raise ValueError("arg: dimension has to be inside the limit dimension")
        
        
        res = np.copy(np.array(self._container, dtype=np.float64))
        res[:, feature_idx] = self.lower_limit[dimension]
        
        return res 

    def shift_to_upper(self, dimension: int, feature_idx: int):
        """_summary_

        Args:
            dimension (int): index from upper limit to state which value to inset into data
            feature_idx (int): index for data to determine in which column to insert the upper limit

        Raises:
            ArgumentError: _description_

        Returns:
            _type_: _description_
        """
        if dimension >= self.dimension:
            raise ValueError("arg: dimension has to be inside the limit dimension")
        
        res = np.copy(np.array(self._container))
        res[:, feature_idx] = np.float64(self.upper_limit[dimension])
        
        return res 
    
    def __repr__(self) -> str:
        data = pd.DataFrame(self.container)
        return f"lower_limit: {self.lower_limit} \n upper_limit: {self.upper_limit} \n container: {np.array(self._container)} \n num_elements: {len(self)} \n"
    
    def __len__(self) -> int:
        return len(self._container)
    

class Limit:
    """The Limit class is the representation of a limit
    It should contain a limit and a dimension what to focus on
    """
    def __init__(self, limit: float, dim: int) -> None:
        self.limit = limit
        self.dim = dim
        

class Limits:
    """Is a class that contains two limits of the same dimension
    """
    def __init__(self, upper_limit: Limit, lower_limit: Limit) -> None:
        assert type(upper_limit) == Limit
        assert type(lower_limit) == Limit
        assert upper_limit.dim == lower_limit.dim
        assert upper_limit.imit >= lower_limit.limit 
        
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.dim = lower_limit.dim
        
    def is_inside(self, point: Iterable):
        if point[self.dim] >= self.lower_limit.limit and point[self.dim] <= self.upper_limit.limit:
            return True
        
        return False