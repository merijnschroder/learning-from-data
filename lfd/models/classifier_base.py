import abc
import logging

from lfd.models.data import Data
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report


class BaseClassifier(abc.ABC):
    '''A base class for a classifier'''

    _classifier_name: str
    _classifier: BaseEstimator
    _is_trained: bool = False

    def train(self, data: Data):
        '''Train the classifier on the training data.'''
        if not hasattr(self._classifier, 'fit'):
            raise TypeError('Invalid classifier: not implementing \'fit\'')
        logging.info('Start training classifier: %s', self.classifier_name)
        self._classifier.fit(data.x_train, data.y_train)  # type: ignore
        self._is_trained = True

    def evaluate_dev(self, data: Data):
        '''Evaluate the classifier on the development set.'''
        logging.info(
            'Start evaluating %s on the development set with %d data points',
            self.classifier_name, len(data.y_dev)
        )
        self._evaluate(data.x_dev, data.y_dev)

    def evaluate_test(self, data: Data):
        '''Evaluate the classifier on the test set.'''
        if not data.has_test_data:
            logging.error('No test data available to evaluate the model with')
            return

        logging.info(
            'Start evaluating %s on the test set with %d data points',
            self.classifier_name, len(data.y_test)
        )
        self._evaluate(data.x_test, data.y_test)

    def _evaluate(self, x_test, y_test):
        '''Evaluate the classifier with the testing data.'''
        if not self._is_trained:
            logging.error('Classifier is not trained, call \'fit\' first.')
            return

        if not hasattr(self._classifier, 'fit'):
            raise TypeError('Invalid classifier: not implementing \'predict\'')

        logging.info('Start evaluating %s on %d data points',
                     self.classifier_name, x_test.shape[0])
        predicted = self._classifier.predict(x_test)  # type: ignore
        print(classification_report(y_test, predicted))

    def _get_classifier_name(self):
        if self._classifier_name == '':
            raise ValueError('Classifier name not set')
        return self._classifier_name

    classifier_name = property(_get_classifier_name)
