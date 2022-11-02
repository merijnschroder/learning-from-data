'''This module contains code that helps with operations on a dataset.'''

import numpy as np
from transformers import AutoTokenizer


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


def lm_encoder(y_data, encoder):
    return encoder.fit_transform(y_data)


def plm_tokenize(x_data, plm_name: str) -> dict:
    '''Tokenize a given set with current PLM tokenizer'''
    tokenizer = AutoTokenizer.from_pretrained(plm_name)

    return tokenizer(
        x_data, padding=True, max_length=100, truncation=True,
        return_tensors="np").data
