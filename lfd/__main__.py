'''This module is the main entry point for the program.'''

import logging
import os
from datetime import datetime

from lfd.models.data import Data


def _set_up_logger():
    '''Configure the logger.'''
    # Create a folder for the current run.
    run_id = datetime.now().strftime('%y%m%d%H%M%S')
    dir_path = f'experiments/{run_id}'
    os.makedirs(dir_path)

    # Create a file and stream handler.
    file_handler = logging.FileHandler(f'{dir_path}/event.log')
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

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


if __name__ == '__main__':
    _set_up_logger()
    logging.info('Start program')
    data = Data('data/train.tsv', 'data/dev.tsv', 'data/test.tsv')
