from typing import TYPE_CHECKING, Tuple, Union
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from mlassistant.core.data import ContentLoader
if TYPE_CHECKING:
    from .mnist_config import MnistConfig


class MnistLoader(ContentLoader):
    """ The MnistLoader class """

    def __init__(self, conf: 'MnistConfig', prefix_name: str, data_specification: str):
        super().__init__(conf, prefix_name, data_specification)
        self._x, self._y = self._load_data(data_specification)

    def _load_data(self, data_specification: str) -> Tuple[np.ndarray, np.ndarray]:
        mnist = fetch_openml('mnist_784')
        x = mnist.data.reshape(70000, 1, 28, 28) / 255.
        y = mnist.target.astype(int)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=17)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=17)

        # Note: due to fixing random state, this block will always result in the same train, val, test split
        # You should always make sure your splits are constant and non-overlapping

        data = {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test': (x_test, y_test),
        }
        return data[data_specification]

    def get_samples_names(self):
        ''' sample names must be unique, they can be either scan_names or scan_dirs.
        Decided to put scan_names. No difference'''
        return [str(i) for i in range(len(self._x))]

    def get_samples_labels(self):
        return self._y

    def reorder_samples(self, indices, new_names):
        self._x = self._x[indices]
        self._y = self._y[indices]

    def get_views_indices(self):
        return self.get_samples_names(),\
            np.arange(len(self._x)).reshape((len(self._x), 1))

    def get_samples_batch_effect_groups(self):
        pass

    def get_placeholder_name_to_fill_function_dict(self):
        """ Returns a dictionary of the placeholders' names (the ones this content loader supports)
        to the functions used for filling them. The functions must receive as input data_loader,
        which is an object of class data_loader that contains information about the current batch
        (e.g. the indices of the samples, or if the sample has many elements the indices of the chosen
        elements) and return an array per placeholder name according to the receives batch information.
        IMPORTANT: Better to use a fixed prefix in the names of the placeholders to become clear which content loader
        they belong to! Some sort of having a mark :))!"""
        return {
            'x': self._get_x,
            'y': self._get_y,
        }

    def _get_x(self, samples_inds: np.ndarray, samples_elements_inds: Union[None, np.ndarray])\
            -> np.ndarray:
        return self._x[samples_inds]

    def _get_y(self, samples_inds: np.ndarray, samples_elements_inds: Union[None, np.ndarray])\
            -> np.ndarray:
        return self._y[samples_inds]