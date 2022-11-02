import abc
import logging
import os
import pickle
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from transformers import PreTrainedModel
from lfd import RUN_ID
from lfd.models.data import Data

class BaseClassifier(abc.ABC):
    '''A base class for a classifier'''

    classifier_id: str
    _classifier_name: str
    _classifier: BaseEstimator | PreTrainedModel
    _is_trained: bool = False

    def __init__(self):
        self.classifier_id = self.classifier_name + '_' + \
            datetime.now().strftime('%y%m%d%H%M%S')

    def train(self, data: Data):
        '''Train the classifier on the training data.'''
        if not hasattr(self._classifier, 'fit'):
            raise TypeError('Invalid classifier: not implementing \'fit\'')

        logging.info('Start training classifier: %s', self.classifier_name)
        self._train(data)
        self._is_trained = True

        # Save the classifier as a file.
        file_path = f'{self.results_path}/model.pkl'
        logging.info('Save trained classifier to %s', file_path)
        with open(file_path, 'wb') as file:
            pickle.dump(self._classifier, file)

    @abc.abstractmethod
    def evaluate_dev(self, data: Data):
        '''Evaluate the classifier on the development set.'''

    @abc.abstractmethod
    def evaluate_test(self, data: Data):
        '''Evaluate the classifier on the test set.'''

    @abc.abstractmethod
    def grid_search(self, data: Data):
        '''Perform a grid-search on the training dataset.'''

    @abc.abstractmethod
    def _train(self, data):
        '''Perform a specialized fit on a model depending on its type'''

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

        predicted = self._evaluation_prediction(x_test)
        report = classification_report(y_test, predicted)
        conf_matrix = confusion_matrix(y_test, predicted)

        file_path = f'{self.results_path}/report.txt'
        logging.info('Writing classification report to %s', file_path)
        with open(file_path, 'a', encoding='utf-8') as file:
            file.writelines(report)

        file_path = f'{self.results_path}/confusion_matrix.json'
        conf_matrix_list = conf_matrix.tolist()
        logging.info('Writing confusion matrix to %s', file_path)
        with open(file_path, 'a', encoding='utf-8') as file:
            file.writelines('[')
            for row in conf_matrix_list:
                if row == conf_matrix_list[-1]:
                    file.writelines(f'\n  {row}')
                else:
                    file.writelines(f'\n  {row},')
            file.writelines('\n]')

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

    def evaluate_dev(self, data: Data):
        self._evaluate(data.get_x_dev(), data.get_y_dev())

    def evaluate_test(self, data: Data):
        self._evaluate(data.get_x_test(), data.get_y_test())

    def _train(self, data):
        self._classifier.fit(  # type: ignore
            data.get_x_train(), data.get_y_train())

    def _evaluation_prediction(self, x_test):
        logging.info('Start evaluating %s on %d data points',
                     self.classifier_name, x_test.shape[0])
        return self._classifier.predict(x_test)  # type: ignore

    def _grid_search_fitting(self, grid_search, data):
        grid_search.fit(data.get_x_train(), data.get_y_train())
