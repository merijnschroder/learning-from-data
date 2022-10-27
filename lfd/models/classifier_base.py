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

import tensorflow as tf


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
        self._train_fitting(data)
        self._is_trained = True

        # Save the classifier as a file.
        file_path = f'{self.results_path}/model.pkl'
        logging.info('Save trained classifier to %s', file_path)
        with open(file_path, 'wb') as file:
            pickle.dump(self._classifier, file)

    @abc.abstractmethod
    def _train_fitting(self, data):
        '''Perform a specialized fit on a model depending on its type'''

    @abc.abstractmethod
    def evaluate_dev(self, data: Data):
        '''Evaluate the classifier on the development set.'''
        logging.info(
            'Start evaluating %s on the development set with %d data points',
            self.classifier_name, len(data.y_dev)
        )

    @abc.abstractmethod
    def evaluate_test(self, data: Data):
        '''Evaluate the classifier on the test set.'''
        if not data.has_test_data:
            logging.error('No test data available to evaluate the model with')
            return

        logging.info(
            'Start evaluating %s on the test set with %d data points',
            self.classifier_name, len(data.y_test)
        )

    @abc.abstractmethod
    def grid_search(self, data: Data):
        '''Perform a grid-search on the training dataset.'''

    def _grid_search(self, data: Data, param_grid: dict[str, list]):
        logging.info('Starting grid-search for %s', self.classifier_name)

        # Perform the grid-search.
        grid_search = GridSearchCV(
            self._classifier, param_grid, scoring='f1_weighted', n_jobs=1,
            refit=False, verbose=10
        )
        self._grid_search_fitting(grid_search, data)

        # Write the results to a file.
        results_file = f'{self.results_path}/grid-search.txt'
        logging.info('Writing results to %s', results_file)
        results = grid_search.cv_results_
        scores = zip(results['params'], results['mean_test_score'])
        with open(results_file, 'w', encoding='utf-8') as file:
            file.writelines(str(s) + '\n' for s in scores)
            file.write(f'\nBest parameters: {grid_search.best_params_} - '
                       f'{grid_search.best_score_}')

    @abc.abstractmethod
    def _grid_search_fitting(self, grid_search, data):
        '''Grid-search for the data for a specific model type'''

    def _evaluate(self, x_test, y_test):
        '''Evaluate the classifier with the testing data.'''
        if not self._is_trained:
            logging.error('Classifier is not trained, call \'fit\' first.')
            return

        if not hasattr(self._classifier, 'fit'):
            raise TypeError('Invalid classifier: not implementing \'predict\'')

        logging.info('Start evaluating %s on %d data points',
                     self.classifier_name, x_test.shape[0])
        predicted = self._evaluation_prediction(x_test)  # type: ignore
        report = classification_report(y_test, predicted)

        file_path = f'{self.results_path}/report.txt'
        logging.info('Writing classification report to %s', file_path)
        with open(file_path, 'a', encoding='utf-8') as file:
            file.writelines(report)

    @abc.abstractmethod
    def _evaluation_prediction(self, x_test):
        '''Calculate the prediction for the specific model'''

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


class BaseBasicClassifier(BaseClassifier):
    '''A base class for a Basic (non-neural network) classifier'''

    classifier_id: str
    _classifier_name: str
    _classifier: BaseEstimator
    _is_trained: bool = False

    def _train_fitting(self, data):
        self._classifier.fit(data.x_train, data.y_train)  # type: ignore

    def evaluate_dev(self, data: Data):
        self._evaluate(data.x_dev, data.y_dev)

    def evaluate_test(self, data: Data):
        self._evaluate(data.x_test, data.y_test)

    def _evaluation_prediction(self, x_test):
        return self._classifier.predict(x_test)  # type: ignore

    def _grid_search_fitting(self, grid_search, data):
        grid_search.fit(data.x_train, data.y_train)


class BaseLMClassifier(BaseClassifier):
    '''A base class for a pre-trained language model classifier'''

    classifier_id: str
    _classifier_name: str
    _classifier: BaseEstimator
    _is_trained: bool = False

    def _train_fitting(self, data):
        validation_data = (data.tokens_dev, data.y_dev_bin)

        self._classifier.fit(data.tokens_train,
                             data.y_train_bin,
                             verbose=self.training_verbosity,
                             epochs=1,
                             batch_size=16,
                             validation_data=validation_data)

    def evaluate_dev(self, data: Data):
        self._evaluate(data.tokens_dev, data.y_dev_bin)

    def evaluate_test(self, data: Data):
        self._evaluate(data.tokens_test, data.y_test_bin)

    def _evaluation_prediction(self, x_test):
        output = self._classifier.predict(x_test)["logits"]
        return tf.round(tf.nn.sigmoid(output))

    def _grid_search_fitting(self, grid_search, data):
        grid_search.fit(data.tokens_train, data.y_train_bin)