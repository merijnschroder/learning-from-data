'''This module is the main entry point for the program.'''

import logging
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer

from lfd import RUN_ID
from lfd.helpers import args_helper
from lfd.models.classifier_base import BaseClassifier
from lfd.models.classifier_knn import KNearestNeighboursClassifier
from lfd.models.classifier_nb import NaiveBayesClassifier
from lfd.models.classifier_randomforest import RandomForestClassifier
from lfd.models.classifier_svc import SupportVectorClassifier
from lfd.models.data import Data, DataLSTM


def _set_up_logger(verbose: bool) -> None:
    '''Configure the logger.'''
    # Create a folder for the current run.
    dir_path = f'experiments/{RUN_ID}'
    os.makedirs(dir_path)

    # Create a file and stream handler.
    file_handler = logging.FileHandler(f'{dir_path}/event.log')
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()

    if verbose:
        stream_handler.setLevel(logging.INFO)
    else:
        stream_handler.setLevel(logging.WARNING)

    # Set the logging format.
    log_format = '%(asctime)s [%(levelname)s] - %(message)s (%(module)s)'
    time_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, time_format)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the file and stream handler to the logger.
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            file_handler,
            stream_handler
        ]
    )


def _train_classifier(classifier: BaseClassifier, data: Data) -> None:
    logging.info('Running %s', classifier.classifier_name)
    classifier.train(data)
    classifier.evaluate_dev(data)
    if data.has_test_data:
        classifier.evaluate_test(data)


def _main():
    args = args_helper.parse_arguments()
    _set_up_logger(args.verbose)
    logging.info('Starting program (id: %s)', RUN_ID)

    # Check if the command line arguments are valid.
    if not args_helper.is_valid(args):
        sys.exit(1)

    # Load the data.
    if args.model == 'lstm':
        data = DataLSTM(args.train_data, args.dev_data, args.test_data)
    else:
        data = Data(args.train_data, args.dev_data, args.test_data)
        if args.vectorizer == 'tfidf':
            data.vectorizer = TfidfVectorizer

    if args.train or args.grid_search:
        classifiers: list[BaseClassifier]
        if args.all_models:
            logging.info('Running all models')
            classifiers = [
                KNearestNeighboursClassifier(), NaiveBayesClassifier(),
                RandomForestClassifier(), SupportVectorClassifier()
            ]
        else:
            classifier = args_helper.get_classifier(args, data)
            if classifier is None:
                sys.exit(1)
            classifiers = [classifier]

        for classifier in classifiers:
            if args.train:
                _train_classifier(classifier, data)
            if args.grid_search:
                classifier.grid_search(data)
    elif args.print_dataset_statistics:
        data.print_statistics()


if __name__ == '__main__':
    _main()
