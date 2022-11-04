from typing import Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier as SKClassifier
from lfd import RANDOM_STATE
from lfd.models.classifier_base import BaseBasicClassifier
from lfd.models.data import Data


class RandomForestClassifier(BaseBasicClassifier):
    '''A Random Forest classifier'''

    _classifier_name: str = 'RandomForest'
    _classifier: BaseEstimator

    def __init__(
        self,
        n_estimators: int = 5,
        criterion: str = 'entropy',
        max_depth: Optional[int] = 100,
        min_samples_leaf: int = 5,
        max_leaf_nodes: Optional[int] = 100
    ):
        self._classifier = SKClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            random_state=RANDOM_STATE
        )
        super().__init__()

    def grid_search(self, data: Data):
        param_grid = {
            'n_estimators': [5, 100, 1000],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [5, 100, 1000],
            'min_samples_leaf': [5, 100],
            'max_leaf_nodes': [100, 50, 5]
        }
        self._grid_search(data, param_grid)
