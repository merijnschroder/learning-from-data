import abc
import logging
import os
import pickle
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from lfd import RUN_ID
from lfd.models.data import Data


class BaseClassifier(abc.ABC):
    '''A base class for a classifier'''

    classifier_id: str
    _classifier_name: str
    _classifier: BaseEstimator
    _is_trained: bool = False

    def __init__(self):
        self.classifier_id = self.classifier_name + '_' + \
            datetime.now().strftime('%y%m%d%H%M%S')

    def train(self, data: Data):
        '''Train the classifier on the training data.'''
        if not hasattr(self._classifier, 'fit'):
            raise TypeError('Invalid classifier: not implementing \'fit\'')

        logging.info('Start training classifier: %s', self.classifier_name)
        self._classifier.fit(data.x_train, data.y_train)  # type: ignore
        self._is_trained = True

        # Save the classifier as a file.
        file_path = f'{self.results_path}/model.pkl'
        logging.info('Save trained classifier to %s', file_path)
        with open(file_path, 'wb') as file:
            pickle.dump(self._classifier, file)

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

    @abc.abstractmethod
    def grid_search(self, data: Data):
        '''Perform a grid-search on the training dataset.'''

    def _grid_search(self, data: Data, param_grid: dict[str, list]):
        logging.info('Starting grid-search for %s', self.classifier_name)

        # Perform the grid-search.
        grid_search = GridSearchCV(self._classifier, param_grid)
        grid_search.fit(data.x_train, data.y_train)

        # Write the results to a file.
        results_file = f'{self.results_path}/grid-search.txt'
        logging.info('Writing results to %s', results_file)
        results = grid_search.cv_results_
        scores = zip(results['params'], results['mean_test_score'])
        with open(results_file, 'w', encoding='utf-8') as file:
            file.writelines(str(s) + '\n' for s in scores)
            file.write(f'\nBest parameters: {grid_search.best_params_} - '
                       f'{grid_search.best_score_}')

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
        report = classification_report(y_test, predicted)

        file_path = f'{self.results_path}/report.txt'
        logging.info('Writing classification report to %s', file_path)
        with open(file_path, 'a', encoding='utf-8') as file:
            file.writelines(report)

    def _get_classifier_name(self) -> str:
        if self._classifier_name == '':
            raise ValueError('Classifier name not set')
        return self._classifier_name

    def _get_results_path(self) -> str:
        path = f'experiments/{RUN_ID}/classifiers/{self._classifier_name}'
        os.makedirs(path, exist_ok=True)
        return path

    classifier_name = property(_get_classifier_name)
    results_path = property(_get_results_path)
