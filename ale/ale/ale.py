import logging
from typing import Any, Callable, Iterable, Tuple
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy import ndarray
from scipy.integrate import simps

from ale.utils.bucket_operations import (
    BinMode,
    get_1d_bucket_index,
    get_linear_buckets,
    get_percentiles,
    sort_into_1d_buckets,
    sort_into_2d_buckets,
)
from ale.utils.data_operations import get_mids, shift_feature_to_value
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator


class ALE:
    def __init__(
        self,
        models: Iterable[object],
        num_buckets: int = 10,  # will be ignored if you want to look at categorical data
        centered: bool = False,
        bin_mode: str = BinMode.percentiles.name,  # will be ignored if you want to look at categorical data
    ):
        self._models = models
        self._num_buckets = num_buckets
        self._centered = centered

        self._bin_mode: str = bin_mode
        self._set_bin_mode(bin_mode)

    def __call__(
        self, X: ndarray, columns: Tuple[int], order: ndarray = []
    ) -> Tuple[ndarray, ndarray]:
        match len(columns):
            case 1:
                return self._1d(X, columns[0], order)
            case 2:
                return self._2d(X, columns, order)
            case _:
                raise ValueError(
                    f"No implemented method for {len(columns)} many features"
                )

    def plot_2d(
        self, scores: ndarray, bins: ndarray, resolution: int = 50, cmap: str = "viridis"
    , color_bar: bool = False, aspect: str = None) -> Tuple[Figure, Axes]:
        mean_score = np.mean(scores, axis=0)
        mids = get_mids(bins.T).T
        interp = RegularGridInterpolator(mids, mean_score)

        axis = np.linspace(mids[:, 0], mids[:, -1], resolution)
        grid = np.stack(np.meshgrid(*axis.T), axis=2)
        inter = interp(grid)

        fig, ax = plt.subplots()
        extent = [mids[0, 0], mids[0, -1], mids[1, -1], mids[1, 0]]
        img = ax.imshow(inter, extent=extent, cmap=mpl.colormaps[cmap], aspect=aspect)
        ax.set_xticks(mids[0])
        ax.set_yticks(mids[1])
        ax.set_xlim([mids[0, 0], mids[0, -1]])
        ax.set_ylim([mids[1, 0], mids[1, -1]])

        if color_bar:
            fig.colorbar(img)

        return fig, ax
        
    def _get_1d_buckets(
        self, X: ndarray, column: int, order: ndarray = []
    ) -> Tuple[ndarray, ndarray]:
        # CATEGORICAL
        if len(order):
            logging.info("categorical buckets")
            num_bins = len(order) - 1
            packed_data = [[]] * num_bins
            for idx in range(num_bins):
                packed_data[idx] = X[
                    np.where(
                        np.logical_or(
                            X[:, column] == order[idx], X[:, column] == order[idx + 1]
                        )
                    )
                ]
            bins = order
            return packed_data, bins

        # CONTINUOUS
        bucket_index, bins = get_1d_bucket_index(
            data=X[:, column],
            bucket_func=self._bucket_func,
            num_buckets=self._num_buckets,
        )
        packed_data = sort_into_1d_buckets(X, bucket_index)
        return packed_data, bins

    def _get_2d_buckets(
        self, X: ndarray, columns: Tuple[int, int], order: ndarray = []
    ) -> Tuple[ndarray, ndarray]:
        # CATEGORICAL
        if len(order):
            raise NotImplementedError

        # CONTINUOUS
        packed_data, bins = sort_into_2d_buckets(
            X, columns, self._bucket_func, self._num_buckets
        )
        return packed_data, bins

    def _model_score_1d(
        self, model: object, data: ndarray, bins: ndarray, column: int
    ) -> ndarray:
        scores = np.empty(len(data))
        for bucket_idx, bucket_content in enumerate(data):
            # if there is no bucket continue
            if len(bucket_content) == 0:
                scores[bucket_idx] = 0
                continue
            # shift data
            content_lower = shift_feature_to_value(
                bucket_content, column, bins[bucket_idx]
            )
            content_upper = shift_feature_to_value(
                bucket_content, column, bins[bucket_idx + 1]
            )

            # predict
            y_lower = model.predict(content_lower)
            y_upper = model.predict(content_upper)

            scores[bucket_idx] = np.mean(y_upper - y_lower)

        scores = np.cumsum(scores)

        return scores

    def _1d(
        self, X: ndarray, column: int, order: ndarray = []
    ) -> Tuple[ndarray, ndarray]:
        packed_data, bins = self._get_1d_buckets(X, column, order)

        scores = np.empty((len(self._models), len(packed_data)))
        for model_idx, model in tqdm(enumerate(self._models)):
            scores[model_idx] = self._model_score_1d(model, packed_data, bins, column)

        if self._centered:
            scores = self._center_ale(scores, bins)

        return scores, bins

    def _model_score_2d(self, model, packed_data, bins, columns):
        scores = np.empty((self._num_buckets, self._num_buckets))
        for row_idx, column in enumerate(packed_data):
            for column_idx, cell in enumerate(column):
                if len(cell) == 0:
                    scores[row_idx, column_idx] = 0
                    continue

                # shift data
                upper_data = shift_feature_to_value(
                    cell.copy(), columns[1], bins[1, row_idx + 1]
                )
                lower_data = shift_feature_to_value(
                    cell.copy(), columns[1], bins[1, row_idx]
                )

                upper_right_data = shift_feature_to_value(
                    upper_data.copy(), columns[0], bins[0, column_idx + 1]
                )
                upper_left_data = shift_feature_to_value(
                    upper_data.copy(), columns[0], bins[0, column_idx]
                )

                lower_right_data = shift_feature_to_value(
                    lower_data.copy(), columns[0], bins[0, column_idx + 1]
                )
                lower_left_data = shift_feature_to_value(
                    lower_data.copy(), columns[0], bins[0, column_idx]
                )

                # predict values
                upper_right_pred = model.predict(upper_right_data)
                upper_left_pred = model.predict(upper_left_data)
                lower_right_pred = model.predict(lower_right_data)
                lower_left_pred = model.predict(lower_left_data)

                # compute score
                scores[row_idx, column_idx] = np.mean(
                    (upper_right_pred - upper_left_pred)
                    - (lower_right_pred - lower_left_pred)
                )

        scores = np.cumsum(scores, axis=0)
        scores = np.cumsum(scores, axis=1)

        return scores

    def _2d(
        self, X: ndarray, columns: Tuple[int], order: Iterable = []
    ) -> Tuple[ndarray, ndarray]:
        packed_data, bins = self._get_2d_buckets(X, columns, order)
        scores = np.empty(
            (len(self._models), self._num_buckets, self._num_buckets), dtype=np.float64
        )
        for model_idx, model in tqdm(enumerate(self._models)):
            scores[model_idx] = self._model_score_2d(model, packed_data, bins, columns)

        if self._centered:
            raise NotImplementedError

        return scores, bins

    @staticmethod
    def _center_ale(scores: ndarray, percentiles: ndarray):
        if len(list(filter(lambda x: isinstance(x, str), percentiles))):
            integral = np.sum(scores, axis=1)
        else:
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

    @property
    def _bucket_func(self) -> Callable:
        if self._bin_mode == BinMode.percentiles.name:
            bucket_func = get_percentiles

        elif self._bin_mode == BinMode.linear.name:
            bucket_func = get_linear_buckets
        else:
            logging.fatal(f"{self._bin_mode=} is not implemented")
        return bucket_func
