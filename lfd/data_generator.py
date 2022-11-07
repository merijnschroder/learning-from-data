'''
This module contains code for adapting a dataset by adding offensive terms to
inoffensive text.
'''

import logging
import os
import pickle
import random
import sys
from typing import List

import numpy as np

from lfd.models.data import Data


def generate_dataset(model_path: str, data: Data) -> None:
    '''
    Adapt the dataset by introducing the best predictors for offensive terms of
    a model to inoffensive tweets.
    '''
    logging.info("Starting dataset generation")

    # Load the model.
    logging.info('Loading the model')
    with open(model_path, 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    # Determine the most offensive terms.
    logging.info('Determining the most offensive terms')
    offensive_terms = _get_most_offensive_terms(model, data)

    # Randomly add offensive terms to inoffensive tweets.
    logging.info('Adding offensive terms to inoffensive tweets')
    train_text = _add_offensive_terms(offensive_terms, data.train_text)
    dev_text = _add_offensive_terms(offensive_terms, data.dev_text)
    test_text = _add_offensive_terms(offensive_terms, data.test_text)

    # Write the data to a file.
    destination_path = 'data/generated'
    os.makedirs(destination_path, exist_ok=True)
    _save_data(train_text, data.get_y_train(), f'{destination_path}/train.tsv')
    _save_data(dev_text, data.get_y_dev(), f'{destination_path}/dev.tsv')
    _save_data(test_text, data.get_y_test(), f'{destination_path}/test.tsv')


def _get_most_offensive_terms(model, data: Data) -> List[str]:
    '''Determine the 100 most offensive terms of the model.'''
    if not hasattr(model, 'coef_'):
        logging.error('Model has no attribute \'coef_\'')
        sys.exit(1)

    # Get the 100 terms with the greatest coefficients.
    features = data.vectorizer.get_feature_names_out()
    indices = np.argsort(model.coef_.data)[0:100]
    return [features[i] for i in indices]


def _add_offensive_terms(offensive_terms: List[str], tweets: List[str]
                         ) -> List[str]:
    '''
    Randomly select an offensive term from the list of offensive terms and add
    it in a random position in each inoffensive tweet.
    '''
    generated_tweets: List[str] = []

    for tweet in tweets:
        # Select a random offensive term.
        offensive_term = random.choice(offensive_terms)

        # Select a random position in the tweet.
        terms = tweet.split(' ')
        location = random.randint(0, len(terms))

        # Insert the randomly selected offensive term at the randomly selected
        # location.
        terms.insert(location, offensive_term)
        generated_tweets.append(' '.join(terms))

    return generated_tweets


def _save_data(tweets: List[str], labels: List[bool], file_path: str) -> None:
    '''Write the data to a file.'''
    logging.info('Writing data to %s', file_path)
    with open(file_path, 'w', encoding='utf-8') as file:
        for i, tweet in enumerate(tweets):
            if labels[i] == True:
                label = 'OFF'
            else:
                label = 'NOT'
            file.write(f'{tweet}\t{label}\n')
