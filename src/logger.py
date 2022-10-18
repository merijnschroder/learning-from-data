'''This module contains code related to logging.'''

import logging
import os
from datetime import datetime


class Logger():
    '''An object that can be used for logging.'''
    _event_logger: logging.Logger

    def __init__(self):
        run_id = datetime.now().strftime('%y%m%d%H%M%S')
        experiment_dir = _create_experiments_folder(run_id)
        self._event_logger = _create_event_logger(experiment_dir)

    def log_event(self, msg: str):
        '''Log an event.'''
        self._event_logger.info(msg)

    def log_warning(self, msg: str):
        '''Log a warning.'''
        self._event_logger.warning(msg)

    def log_error(self, msg: str):
        '''Log an error.'''
        self._event_logger.error(msg)


def _create_experiments_folder(run_id: str) -> str:
    dir_path = f'experiments/{run_id}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def _create_event_logger(experiment_dir: str) -> logging.Logger:
    # Create the logger.
    event_logger = logging.getLogger('event')
    event_logger.setLevel(logging.INFO)

    # Create a file and stream handler.
    file_path = f'{experiment_dir}/event.log'
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)

    # Set the logging format.
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    time_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, time_format)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the file and stream handler to the logger.
    event_logger.addHandler(file_handler)
    event_logger.addHandler(stream_handler)

    return event_logger
