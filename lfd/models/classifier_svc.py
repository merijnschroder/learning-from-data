from typing import Literal

from sklearn import svm
from sklearn.base import BaseEstimator
from lfd import RANDOM_STATE
from lfd.models.classifier_base import BaseClassifier
from lfd.models.data import Data


class SupportVectorClassifier(BaseClassifier):
    '''A Support Vector Classifier'''

    _classifier_name: str = 'SupportVectorClassifier'
    _classifier: BaseEstimator

    def __init__(
        self,
        C: float = 1,
        kernel: Literal['linear', 'poly', 'rbf', 'sigmoid',
                        'precomputed'] = 'rbf'
    ):
        self._classifier = svm.SVC(C=C, kernel=kernel,
                                   random_state=RANDOM_STATE)
        super().__init__()

    def grid_search(self, data: Data):
        param_grid = {
            'C': [0.1, 0.5, 1, 5, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        self._grid_search(data, param_grid)
