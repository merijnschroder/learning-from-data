'''This module contains the Data class.'''

import logging
import os
from typing import Type

import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
from numpy.typing import NDArray
from scipy.sparse import csr_matrix as sparse_row_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from lfd.helpers.data_helper import plm_tokenize, print_label_statistics


class Data:
    '''This class holds all data.'''

    _y_train: list[bool]
    _y_dev: list[bool]
    _y_test: list[bool] = []

    _train_text: list[str]
    _dev_text: list[str]
    _test_text: list[str]

    _vocabulary: NDArray
    _vectorizer: CountVectorizer

    def __init__(self, train_file: str, dev_file: str, test_file: str):
        # Load the data from the data files.
        logging.info('Start loading data')
        self._train_text, self._y_train = self._read_data_from_file(train_file)
        self._dev_text, self._y_dev = self._read_data_from_file(dev_file)
        self._test_text = []
        if test_file is not None:
            self._test_text, self._y_test = self._read_data_from_file(
                test_file)
        self._default_vectorizer()

    def _default_vectorizer(self):
        '''Set the default vectorizer.'''
        logging.info('Using the default Vectorization pipeline')
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
        print_label_statistics(self._y_train + self._y_dev + self._y_test)

        print('\nTrain')
        print_label_statistics(self._y_train)

        print('\nDev')
        print_label_statistics(self._y_dev)

        print('\nTest')
        print_label_statistics(self._y_test)

    def get_x_train(self, plm_name: str = ''):
        return self._transform_text(self._train_text, plm_name)

    def get_x_dev(self, plm_name: str = ''):
        return self._transform_text(self._dev_text, plm_name)

    def get_x_test(self, plm_name: str = ''):
        return self._transform_text(self._test_text, plm_name)

    def get_y_train(self, encoded: bool = False):
        return self._transform_labels(self._y_train, encoded)

    def get_y_dev(self, encoded: bool = False):
        return self._transform_labels(self._y_dev, encoded)

    def get_y_test(self, encoded: bool = False):
        return self._transform_labels(self._y_test, encoded)

    def _get_vocabulary(self):
        voc = self._vectorizer.vocabulary_
        return list(voc.keys())

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

    def _transform_text(self, text: list[str], plm_name: str):
        if plm_name == '':
            return self._vectorize_text(text)
        else:
            return self._tokenize_text(text, plm_name)

    def _transform_labels(self, labels: list[bool], encoded: bool):
        if encoded:
            encoder = LabelBinarizer()
            return encoder.fit_transform(labels)
        else:
            return labels

    def _vectorize_text(self, text: list[str]) -> sparse_row_matrix:
        return self._vectorizer.transform(text)

    def _tokenize_text(self, text: list[str], plm_name: str) -> dict:
        return plm_tokenize(text, plm_name)

    def _set_vectorizer(self, vectorizer: Type[CountVectorizer]):
        logging.info('Setting custom vectorizer: %s', vectorizer.__name__)
        self._vectorizer = vectorizer()
        self._vectorizer.fit(self._vocabulary)

    def _has_test_data(self) -> bool:
        return len(self._y_test) > 0

    vectorizer = property(fset=_set_vectorizer)
    has_test_data = property(fget=_has_test_data)
    voc = property(fget=_get_vocabulary)


class DataLSTM(Data):
    def _default_vectorizer(self):
        logging.info('Using LSTM Vectorization pipeline')
        self._text_ds = tf.data.Dataset.from_tensor_slices(
            self._train_text + self._dev_text)
        self._vocabulary = np.unique(
            self._train_text + self._dev_text + self._test_text)
        self._vectorizer = TextVectorization(
            standardize=None, output_sequence_length=50)
        self._vectorizer.adapt(self._text_ds)
        self._vocabulary = self._vectorizer.get_vocabulary()

    def _get_vocabulary(self):
        return self._vocabulary

    def _transform_text(self, text: list[str], plm_name: str):
        return self._vectorizer(np.array([[s] for s in text])).numpy()

    voc = property(fget=_get_vocabulary)
