from typing import Any, Tuple
import numpy as np
from numpy import ndarray

from ale.utils.bucket_operations import get_1d_bucket_index, sort_into_buckets
from ale.utils.data_operations import shift_column_to_value


class ALE:
    def __init__(self, model, num_buckets: int):
        self._model = model
        self._num_buckets = num_buckets

    def __call__(self, X: ndarray, columns: Tuple[int]) -> Tuple[ndarray, ndarray]:
        match len(columns):
            case 1:
                return self._1d(X, columns[0])
            case 2:
                self._2d(X, columns)
            case _:
                raise ValueError(
                    f"No implemented method for {len(columns)} many features"
                )

    def plot(self):
        pass

    def _1d(self, X: ndarray, column: int, centered: bool = True) -> Tuple[ndarray, ndarray]:
        bucket_index, bins = get_1d_bucket_index(X[:, column], self._num_buckets)
        packed_data = sort_into_buckets(X, bucket_index)

        score = np.empty(self._num_buckets)
        for bucket_idx, bucket_content in enumerate(packed_data):
            # predictions for lower bound 
            content_lower = shift_column_to_value(
                bucket_content, column, bins[bucket_idx]
            )
            y_lower = self._model.predict(content_lower)
            # prediction for upper bound
            content_upper = shift_column_to_value(
                bucket_content, column, bins[bucket_idx + 1]
            )
            y_upper = self._model.predict(content_upper)

            score[bucket_idx] = np.mean(y_upper - y_lower)
        
        score = np.cumsum(score)
        
        if centered:
            score = self._center_ale(score)
        
        return score, bins 

    def _2d(self, X: ndarray, columns: Tuple[int]):
        raise NotImplementedError
    
    @staticmethod
    def _center_ale(score: ndarray):
        return score - score.mean()
