from typing import Literal

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from lfd.models.classifier_base import BaseClassifier


class KNearestNeighboursClassifier(BaseClassifier):
    '''A K-Nearest Neighbours classifier'''

    _classifier_name: str = 'KNearestNeighbours'
    _classifier: BaseEstimator

    def __init__(
        self,
        n_neighbours: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        distance_metric: Literal['euclidean', 'manhattan', 'cosine',
                                 'haversine', 'minkowski'] = 'minkowski'
    ):
        self.classifier = KNeighborsClassifier(
            n_neighbors=n_neighbours,
            weights=weights,
            metric=distance_metric
        )
