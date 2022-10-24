from typing import Literal, Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier as SKClassifier
from lfd import RANDOM_STATE
from lfd.models.classifier_base import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    '''A Random Forest classifier'''

    _classifier_name: str = 'RandomForest'
    _classifier: BaseEstimator

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_leaf_nodes: Optional[int] = None
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
