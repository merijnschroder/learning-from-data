'''This module contains code that helps with operations on a dataset.'''

import numpy as np


def print_label_statistics(labels: list) -> None:
    '''Print statistics for the list of labels.'''
    print(f'Total: {len(labels)}')
    print_class_distribution(labels)


def print_class_distribution(labels: list) -> None:
    '''Print for each class the the percentage of occurrences.'''
    total_cnt: int = len(labels)
    for label in np.unique(labels):
        label_cnt: int = len(list(filter(
            lambda l, label=label: l == label, labels)))
        print(f'{label}: {label_cnt} ({np.round(label_cnt / total_cnt, 3)})')
