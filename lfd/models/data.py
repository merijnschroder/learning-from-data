'''This module contains the Data class.'''

import logging
import os
from typing import Type

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix as sparse_row_matrix
from sklearn.feature_extraction.text import CountVectorizer
from lfd.helpers.data_helper import print_label_statistics


class Data:
    '''This class holds all data.'''

    y_train: list[bool]
    y_dev: list[bool]
    y_test: list[bool] = []

    _train_text: list[str]
    _dev_text: list[str]
    _test_text: list[str]

    _vocabulary: NDArray
    _vectorizer: CountVectorizer = CountVectorizer()

    def __init__(self, train_file: str, dev_file: str, test_file: str):
        # Load the data from the data files.
        logging.info('Start loading data')
        self._train_text, self.y_train = self._read_data_from_file(train_file)
        self._dev_text, self.y_dev = self._read_data_from_file(dev_file)
        self._test_text = []
        if test_file is not None:
            self._test_text, self.y_test = self._read_data_from_file(test_file)

        # Set the default vectorizer.
        self._vocabulary = np.unique(
            self._train_text + self._dev_text + self._test_text)
        self._vectorizer = CountVectorizer()
        self._vectorizer.fit(self._vocabulary)

    def print_statistics(self):
        '''Print information about the dataset stored in this class.'''
        logging.info('Printing dataset statistics')

        # Print the statistics for all the individual datasets and all datasets
        # combined.
        print('\nClass distribution')

        print('\nFull dataset')
        print_label_statistics(self.y_train + self.y_dev + self.y_test)

        print('\nTrain')
        print_label_statistics(self.y_train)

        print('\nDev')
        print_label_statistics(self.y_dev)

        print('\nTest')
        print_label_statistics(self.y_test)

    def _read_data_from_file(
            self, data_file_path: str) -> tuple[list[str], list[bool]]:
        '''Read the data from the data file.'''
        logging.info('Loading data from %s', data_file_path)
        features: list[str] = []
        labels: list[bool] = []

        # Return empty lists if the file does not exist.
        if data_file_path is None or not os.path.exists(data_file_path):
            logging.warning(
                'Data file %s does not exist, returning empty lists',
                data_file_path
            )
            return features, labels

        # Open the file, parse the data, and return the features and labels.
        with open(data_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = line.strip()
                tweet, label = tokens.split('\t')
                features.append(" ".join(tweet.split()).strip())
                labels.append(label == 'OFF')

        return features, labels

    def _vectorize_text(self, text: list[str]) -> sparse_row_matrix:
        return self._vectorizer.transform(text)

    def _set_vectorizer(self, vectorizer: Type[CountVectorizer]):
        logging.info('Setting custom vectorizer: %s', vectorizer.__name__)
        self._vectorizer = vectorizer()
        self._vectorizer.fit(self._vocabulary)

    def _has_test_data(self) -> bool:
        return len(self.y_test) > 0

    def _get_x_train(self) -> sparse_row_matrix:
        return self._vectorize_text(self._train_text)

    def _get_x_dev(self) -> sparse_row_matrix:
        return self._vectorize_text(self._dev_text)

    def _get_x_test(self) -> sparse_row_matrix:
        return self._vectorize_text(self._test_text)

    vectorizer = property(fset=_set_vectorizer)
    has_test_data = property(fget=_has_test_data)
    x_train = property(fget=_get_x_train)
    x_dev = property(fget=_get_x_dev)
    x_test = property(fget=_get_x_test)
