'''This module contains the Data class.'''

import os

from logger import Logger


class Data:
    '''This class holds all data.'''
    _logger: Logger

    x_train: list
    y_train: list
    x_test: list
    y_test: list
    x_dev: list
    y_dev: list

    def __init__(self, train_file: str, dev_file: str, test_file: str, logger: Logger):
        self._logger = logger

        # Load the data from the data files.
        self.x_train, self.y_train = self._read_data_from_file(train_file)
        self.x_dev, self.y_dev = self._read_data_from_file(dev_file)
        if test_file is not None:
            self.x_test, self.y_test = self._read_data_from_file(test_file)

    def _read_data_from_file(self, file_name: str) -> tuple[list, list]:
        '''Read the data from the data file.'''
        self._logger.log_event(f'Loading data from {file_name}')
        features = []
        labels = []

        # Return empty lists if the file does not exist.
        if not os.path.exists(file_name):
            self._logger.log_warning(f'Data file {file_name} does not exist, '
                                     'returning empty strings')
            return features, labels

        # Open the file, parse the data, and return the features and labels.
        with open(file_name, encoding='utf-8') as file:
            for line in file:
                tokens = line.strip()
                tweet, label = tokens.split('\t')
                features.append(" ".join(tweet.split()).strip())
                labels.append(label)

        return features, labels
