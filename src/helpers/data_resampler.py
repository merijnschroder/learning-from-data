from sklearn.utils import resample

import numpy as np
import os
import pandas as pd

RANDOM_STATE = 1234


def file_writer(data, data_name, operation_name):
    '''
    Checks whether a folder exists for the operation name, then
    saves the data to a file in that folder based on the data name.
    '''
    folder_struct = '../../data/'

    if not os.path.exists(folder_struct + operation_name):
        os.mkdir(folder_struct + operation_name)

    data.to_csv(path_or_buf=f'{folder_struct}{operation_name}/{data_name}.tsv',
                sep='\t', index=False, header=False)


def sampling(data, replace=True, sampling: str = 'down'):
    '''
    Executes a resampling based on the given sampling instruction
    and returns a resampled pandas dataframe.
    '''
    data_not = data[data['Label (y)'] == 'NOT']
    data_off = data[data['Label (y)'] == 'OFF']

    if len(data_not) > len(data_off):
        majority_class = data_not
        minority_class = data_off
    else:
        majority_class = data_off
        minority_class = data_not

    if sampling == 'down':
        changed_label = resample(majority_class,
                                 replace=replace,
                                 n_samples=len(minority_class),
                                 random_state=RANDOM_STATE)

        return pd.concat([changed_label, minority_class]).sort_index()

    else:
        if replace is False:
            print('Setting replace for upsampling does not work: '
                  'upsampling uses replace=True')

        changed_label = resample(minority_class,
                                 replace=True,
                                 n_samples=len(majority_class),
                                 random_state=RANDOM_STATE)

        return pd.concat([changed_label, majority_class]).sort_index()


def main():
    folder_struct = '../../data/'
    names = 'Sentence (x) | Label (y)'.split(' | ')

    train = pd.read_csv(f'{folder_struct}train.tsv',
                        sep='\t',
                        names=names)
    dev = pd.read_csv(f'{folder_struct}dev.tsv',
                      sep='\t',
                      names=names)
    test = pd.read_csv(f'{folder_struct}test.tsv',
                       sep='\t',
                       names=names)

    datasets = {'train': train, 'dev': dev, 'test': test}

    for name, data in datasets.items():
        data_downsampled = sampling(data, replace=False, sampling='down')
        file_writer(data_downsampled, name, 'downsampled')

        data_upsampled = sampling(data, replace=True, sampling='up')
        file_writer(data_upsampled, name, 'upsampled')


if __name__ == '__main__':
    main()
