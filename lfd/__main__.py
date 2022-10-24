'''This module is the main entry point for the program.'''

import argparse
import logging
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer

from lfd import RUN_ID
from lfd.models.classifier_base import BaseClassifier
from lfd.models.data import Data
from lfd.models.classifier_knn import KNearestNeighboursClassifier
from lfd.models.classifier_nb import NaiveBayesClassifier
from lfd.models.classifier_randomforest import RandomForestClassifier
from lfd.models.classifier_svc import SupportVectorClassifier


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


def _parse_arguments() -> argparse.Namespace:
    '''Parse the command line arguments.'''
    parser = argparse.ArgumentParser()

    # Data files
    parser.add_argument(
        '--train-data',
        type=str,
        default='data/original/train.tsv',
        help='The path of the file containing training data '
        '(default: data/original/train.tsv)'
    )
    parser.add_argument(
        '--dev-data',
        type=str,
        default='data/original/dev.tsv',
        help='The path of the file containing development data '
        '(default: data/original/dev.tsv)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        help='The path of the file containing testing data'
    )

    # Run modes
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train and evaluate the specified model (should be accompanied '
             'by --model or --all-models)'
    )
    parser.add_argument('--print-dataset-statistics', action='store_true',
                        help='Print the statistics of the dataset')

    # Options
    parser.add_argument(
        '--model',
        type=str,
        choices=['knn', 'nb', 'randomforest', 'svc'],
        help='The model to train'
    )
    parser.add_argument('--all-models', action='store_true',
                        help='Train and evaluate all models')
    parser.add_argument('--verbose', action='store_true',
                        help='Write logs to the console')

    return parser.parse_args()


def _get_classifier(args: argparse.Namespace) -> BaseClassifier:
    if args.model is None:
        logging.error('Please specify --model')
        sys.exit(1)
    if args.model == 'knn':
        return KNearestNeighboursClassifier()
    if args.model == 'nb':
        return NaiveBayesClassifier()
    if args.model == 'randomforest':
        return RandomForestClassifier()
    if args.model == 'svc':
        return SupportVectorClassifier()
    logging.error('Unknown model %s.', args.model)
    sys.exit(1)


def _run_classifier(classifier: BaseClassifier, data: Data) -> None:
    logging.info('Running %s', classifier.classifier_name)
    classifier.train(data)
    classifier.evaluate_dev(data)
    if data.has_test_data:
        classifier.evaluate_test(data)


def _main():
    args = _parse_arguments()
    _set_up_logger(args.verbose)
    logging.info('Starting program (id: %s)', RUN_ID)

    # Check if the command line arguments are valid.
    if not (args.train or args.print_dataset_statistics):
        logging.error('Either run the program with --train or '
                      '--print-dataset-statistics')
        sys.exit(1)

    # Load the data.
    data = Data(args.train_data, args.dev_data, args.test_data)
    data.vectorizer = TfidfVectorizer

    if args.train:
        if args.all_models:
            logging.info('Running all models')
            classifiers = [
                KNearestNeighboursClassifier(), NaiveBayesClassifier(),  
                NaiveBayesClassifier(), RandomForestClassifier(),
                SupportVectorClassifier()
            ]
            for classifier in classifiers:
                _run_classifier(classifier, data)
        else:
            classifier = _get_classifier(args)
            _run_classifier(classifier, data)
    elif args.print_dataset_statistics:
        data.print_statistics()


if __name__ == '__main__':
    _main()
