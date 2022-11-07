'''
This script takes a confusion matrix .json file as '--input' and prints
an equivalent .png version (in the same folder as the input file)
for publishing.
'''


import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import argparse


def parse_arguments() -> argparse.Namespace:
    '''Parse the command line arguments.'''
    parser = argparse.ArgumentParser()

    # Data files
    parser.add_argument(
        '--input',
        type=str,
        help='The path of the .json file '
        'containing a confusion matrix'
    )

    return parser.parse_args()


def json_importer(file_location) -> list:
    '''Import a .json as an array'''
    with open(file_location, 'r', encoding='utf-8') as f:
        conf_matrix = json.load(f)

    return conf_matrix


def write_confmatrix_graphics(confmatrix: np.ndarray, file_location) -> None:
    '''Save confusion matrix data as a nicely formatted .png image.'''
    labels = 'NOT OFF'.split()
    ticks = range(0, 2)
    ConfusionMatrixDisplay(confmatrix).plot()

    plt.rcParams.update({'font.size': 12})
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.savefig(f'{file_location[:-5]}.png', transparent=True, dpi=300)


def main():
    args = parse_arguments()

    if args.input:
        conf_matrix = np.array(json_importer(args.input))
        write_confmatrix_graphics(conf_matrix, args.input)
    else:
        print(f'\n{"Warning:":<10} missing an \'input\' parameter!'
              '\n\nSee the \'--help\' or \'-h\' section!')


if __name__ == '__main__':
    main()
