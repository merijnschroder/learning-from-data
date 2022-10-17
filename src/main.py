'''This module is the main entry point for the program.'''

from logger import Logger


class Main:
    '''This class ties all modules together to work as one program.'''

    _class_name: str
    _logger: Logger

    def __init__(self):
        # Instantiate the logger and determine the current class name for the
        # logger.
        self._logger = Logger()
        self._class_name = type(self).__name__

    def main(self):
        '''Run the program'''
        self._logger.log_event('Starting program', self._class_name)


if __name__ == '__main__':
    Main().main()
