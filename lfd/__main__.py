'''This module is the main entry point for the program.'''

import argparse
import logging
import os
from datetime import datetime

from lfd.models.data import Data


def _set_up_logger() -> None:
    '''Configure the logger.'''
    # Create a folder for the current run.
    run_id = datetime.now().strftime('%y%m%d%H%M%S')
    dir_path = f'experiments/{run_id}'
    os.makedirs(dir_path)

    # Create a file and stream handler.
    file_handler = logging.FileHandler(f'{dir_path}/event.log')
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)

    # Set the logging format.
    log_format = '%(asctime)s [%(levelname)s] - %(module)s: %(message)s'
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
        default='data/train.tsv',
        help='The path of the file containing training data '
        '(default: data/train.tsv)'
    )
    parser.add_argument(
        '--dev-data',
        type=str,
        default='data/dev.tsv',
        help='The path of the file containing development data '
        '(default: data/dev.tsv)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default='data/test.tsv',
        help='The path of the file containing testing data '
        '(default: data/test.tsv)'
    )

    # Run modes
    parser.add_argument('--print-dataset-statistics', action='store_true',
                        help='Print the statistics of the dataset')

    return parser.parse_args()


if __name__ == '__main__':
    _set_up_logger()
    logging.info('Start program')
    args = _parse_arguments()

    data = Data(args.train_data, args.dev_data, args.test_data)

    if args.print_dataset_statistics:
        data.print_statistics()
