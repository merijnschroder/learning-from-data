from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB
from lfd.models.classifier_base import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    '''A Naive Bayes classifier'''

    _classifier_name: str = 'NaiveBayes'
    _classifier: BaseEstimator

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True):
        self._classifier = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
