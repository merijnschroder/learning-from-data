import numpy as np
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB
from lfd.models.classifier_base import BaseBasicClassifier
from lfd.models.data import Data


class NaiveBayesClassifier(BaseBasicClassifier):
    '''A Naive Bayes classifier'''

    _classifier_name: str = 'NaiveBayes'
    _classifier: BaseEstimator

    def __init__(self, alpha: float = 0.8, fit_prior: bool = True):
        self._classifier = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        super().__init__()

    def grid_search(self, data: Data):
        param_grid = {
            'alpha': list(np.arange(0, 1.1, 0.1)),
            'fit_prior': [True, False]
        }
        self._grid_search(data, param_grid)
