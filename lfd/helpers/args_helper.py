'''This module contains helpers for working with command line arguments.'''
import argparse
import logging
from typing import Union

from lfd.models.classifier_base import BaseClassifier
from lfd.models.classifier_knn import KNearestNeighboursClassifier
from lfd.models.classifier_lm import LanguageModelClassifier
from lfd.models.classifier_lstm import LSTMClassifier
from lfd.models.classifier_nb import NaiveBayesClassifier
from lfd.models.classifier_randomforest import RandomForestClassifier
from lfd.models.classifier_svc import SupportVectorClassifier
from lfd.models.data import Data


def parse_arguments() -> argparse.Namespace:
    '''Parse the command line arguments.'''
    parser = argparse.ArgumentParser()

    # Data files
    parser.add_argument(
        '--train-data',
        type=str,
        default='data/deduplicated/train.tsv',
        help='The path of the file containing training data '
        '(default: data/deduplicated/train.tsv)'
    )
    parser.add_argument(
        '--dev-data',
        type=str,
        default='data/deduplicated/dev.tsv',
        help='The path of the file containing development data '
        '(default: data/deduplicated/dev.tsv)'
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
    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Perform a grid-search on the specified model (should be '
             'accompanied by --model or --all-models) '
             'Note: This does not work with the lstm or PLMs and will '
             'raise an error'
    )
    parser.add_argument(
        '--generate-dataset',
        action='store_true',
        help='Adapt the datasets by introducing offensive terms to'
             'non-offensive tweets (should be accompanied by --model-path)'
    )

    # Options
    parser.add_argument(
        '--model',
        type=str,
        choices=['knn', 'nb', 'randomforest', 'svc', 'lstm', 'plm'],
        help='The type of model to train'
    )
    parser.add_argument(
        '--language-model',
        type=str,
        choices=['GroNLP/hateBERT',
                 'bert-base-cased',
                 'bert-base-uncased',
                 'microsoft/deberta-v3-base'],
        default='bert-base-uncased',
        help='The name of the Pre-trained Language Model (required when '
             'model is set to \'plm\', default: bert-base-uncased)'
    )
    parser.add_argument('--all-models', action='store_true',
                        help='Train and evaluate all models')
    parser.add_argument('--verbose', action='store_true',
                        help='Write logs to the console')
    parser.add_argument(
        '--vectorizer',
        type=str,
        choices=['bag-of-words', 'tfidf'],
        default='bag-of-words',
        help='The type of vectorizer to use for the models that require one '
             '(default: bag-of-words)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='The path of the model from which to use the feature importances '
             'for the dataset generation')

    return parser.parse_args()


def is_valid(args: argparse.Namespace) -> bool:
    '''Determine whether the command line arguments are valid.'''
    if not (args.train or args.grid_search or args.print_dataset_statistics or
            args.generate_dataset):
        logging.error('Either run the program with --train, --grid-search, or '
                      '--print-dataset-statistics')
        return False

    if args.generate_dataset and args.model_path is None:
        logging.error('Specify --model-path when generating a dataset')
        return False

    if args.generate_dataset and args.test_data is None:
        logging.error('Specify --test-data when generating a dataset')
        return False

    if args.model == 'lstm' or args.model == 'plm' and args.grid_search:
        logging.error('Grid search is not implemented for LSTMs or PLMs')
        return False

    return True


def get_classifier(args: argparse.Namespace,
                   data: Data) -> Union[BaseClassifier, None]:
    '''Get the correct classifier based on the command line arguments.'''
    if args.model is None:
        logging.error('Please specify --model')
        return None
    if args.model == 'knn':
        return KNearestNeighboursClassifier()
    if args.model == 'nb':
        return NaiveBayesClassifier()
    if args.model == 'randomforest':
        return RandomForestClassifier()
    if args.model == 'svc':
        return SupportVectorClassifier()
    if args.model == 'lstm':
        return LSTMClassifier(data)
    if args.model == 'plm':
        return LanguageModelClassifier(model_name=args.language_model)
    logging.error('Unknown model %s.', args.model)
    return None
