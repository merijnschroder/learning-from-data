'''This module contains the Data class.'''

import logging
import os

from lfd.helpers.data_helper import print_label_statistics


class Data:
    '''This class holds all data.'''
    x_train: list
    y_train: list
    x_dev: list
    y_dev: list
    x_test: list
    y_test: list

    def __init__(self, train_file: str, dev_file: str, test_file: str):
        # Load the data from the data files.
        logging.info('Start loading data')
        self.x_train, self.y_train = self._read_data_from_file(train_file)
        self.x_dev, self.y_dev = self._read_data_from_file(dev_file)
        if test_file is not None:
            self.x_test, self.y_test = self._read_data_from_file(test_file)

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

    def _read_data_from_file(self, data_file_path: str) -> tuple[list, list]:
        '''Read the data from the data file.'''
        logging.info('Loading data from %s', data_file_path)
        features = []
        labels = []

        # Return empty lists if the file does not exist.
        if not os.path.exists(data_file_path):
            logging.warning(
                'Data file %s does not exist, returning empty strings',
                data_file_path
            )
            return features, labels

        # Open the file, parse the data, and return the features and labels.
        with open(data_file_path, encoding='utf-8') as file:
            for line in file:
                tokens = line.strip()
                tweet, label = tokens.split('\t')
                features.append(" ".join(tweet.split()).strip())
                labels.append(label)

        return features, labels
