from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from lfd.models.classifier_base import BaseBasicClassifier
from lfd.models.data import Data


class KNearestNeighboursClassifier(BaseBasicClassifier):
    '''A K-Nearest Neighbours classifier'''

    _classifier_name: str = 'KNearestNeighbours'
    _classifier: BaseEstimator

    def __init__(
        self,
        n_neighbours: int = 3,
        weights: str = 'distance',
        distance_metric: str = 'euclidean'
    ):
        self._classifier = KNeighborsClassifier(
            n_neighbors=n_neighbours,
            weights=weights,
            metric=distance_metric
        )
        super().__init__()

    def grid_search(self, data: Data):
        param_grid = {
            'n_neighbors': [1, 2, 3, 5, 8, 13, 21, 34, 55],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'cosine', 'haversine',
                       'minkowski']
        }
        self._grid_search(data, param_grid)
