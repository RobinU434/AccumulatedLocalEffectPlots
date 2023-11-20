import logging
from typing import Any, Iterable, Tuple
import numpy as np
from numpy import ndarray
from scipy.integrate import simps

from ale.utils.bucket_operations import (
    BinMode,
    get_1d_bucket_index,
    get_linear_buckets,
    get_percentiles,
    sort_into_1d_buckets,
)
from ale.utils.data_operations import get_mids, shift_column_to_value


class ALE:
    def __init__(
        self,
        models: Iterable[object],
        num_buckets: int,
        centered: bool = False,
        bin_mode: str = BinMode.percentiles.name,
    ):
        self._models = models
        self._num_buckets = num_buckets
        self._centered = centered

        self._bin_mode: str
        self._set_bin_mode(bin_mode)

    def __call__(self, X: ndarray, columns: Tuple[int]) -> Tuple[ndarray, ndarray]:
        match len(columns):
            case 1:
                return self._1d(X, columns[0])
            case 2:
                return self._2d(X, columns)
            case _:
                raise ValueError(
                    f"No implemented method for {len(columns)} many features"
                )

    def plot(self):
        pass

    def _get_1d_buckets(self, X: ndarray, column: int) -> Tuple[ndarray, ndarray]:
        if self._bin_mode == BinMode.percentiles.name:
            bucket_func = get_percentiles

        elif self._bin_mode == BinMode.linear.name:
            bucket_func = get_linear_buckets
        else:
            logging.fata

        bucket_index, bins = get_1d_bucket_index(
            data=X[:, column], bucket_func=bucket_func, num_buckets=self._num_buckets
        )
        packed_data = sort_into_1d_buckets(X, bucket_index)
        return packed_data, bins

    def _model_score_1d(
        self, model: object, data: ndarray, bins: ndarray, column: int
    ) -> ndarray:
        scores = np.empty(self._num_buckets)
        for bucket_idx, bucket_content in enumerate(data):
            # if there is no bucket continue
            if len(bucket_content) == 0:
                scores[bucket_idx] = 0
                continue
            # predictions for lower bound
            content_lower = shift_column_to_value(
                bucket_content, column, bins[bucket_idx]
            )
            y_lower = model.predict(content_lower)
            # prediction for upper bound
            content_upper = shift_column_to_value(
                bucket_content, column, bins[bucket_idx + 1]
            )
            y_upper = model.predict(content_upper)

            scores[bucket_idx] = np.mean(y_upper - y_lower)

        scores = np.cumsum(scores)

        return scores

    def _1d(self, X: ndarray, column: int) -> Tuple[ndarray, ndarray]:
        packed_data, bins = self._get_1d_buckets(X, column)

        scores = np.empty((len(self._models), self._num_buckets))
        for model_idx, model in enumerate(self._models):
            model_score = self._model_score_1d(model, packed_data, bins, column)
            scores[model_idx] = model_score

        if self._centered:
            scores = self._center_ale(scores, bins)

        return scores, bins

    def _2d(self, X: ndarray, columns: Tuple[int]):
        pass

    @staticmethod
    def _center_ale(scores: ndarray, percentiles: ndarray):
        x = get_mids(percentiles)
        integral = np.trapz(scores, x=x, axis=1) / (x[-1] - x[0])
        centered = -integral[:, None] + scores
        return centered

    def _set_bin_mode(self, bin_mode: str):
        if bin_mode not in BinMode._member_map_.keys():
            logging.fatal(
                f"bin_mode has to be in: {list(BinMode._member_map_.keys())}. But it was  '{bin_mode}'"
            )
            return
        self._bin_mode = bin_mode
