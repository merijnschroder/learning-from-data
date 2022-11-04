import os

import pandas as pd
from sklearn.utils import resample

from lfd import RANDOM_STATE


def file_writer(folder_struct, data, data_name, operation_name):
    '''
    Checks whether a folder exists for the operation name, then
    saves the data to a file in that folder based on the data name.
    '''
    if not os.path.exists(folder_struct + operation_name):
        os.mkdir(folder_struct + operation_name)

    data.to_csv(path_or_buf=f'{folder_struct}{operation_name}/{data_name}.tsv',
                sep='\t', index=False, header=False)


def sampling(data, replace=True, sampling_type: str = 'under'):
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

    if sampling_type == 'under':
        changed_label = resample(majority_class,
                                 replace=replace,
                                 n_samples=len(minority_class),
                                 random_state=RANDOM_STATE)

        return pd.concat([changed_label, minority_class]).sort_index()

    else:
        if replace is False:
            print('Setting replace for oversampling does not work: '
                  'oversampling uses replace=True')

        changed_label = resample(minority_class,
                                 replace=True,
                                 n_samples=len(majority_class),
                                 random_state=RANDOM_STATE)

        return pd.concat([changed_label, majority_class]).sort_index()


def main():
    duplicates = True
    folder_struct = '../../data/'
    names = 'Sentence (x) | Label (y)'.split(' | ')
    datasets = {}

    datasets['train'] = pd.read_csv(f'{folder_struct}train.tsv',
                                    sep='\t',
                                    names=names)
    datasets['dev'] = pd.read_csv(f'{folder_struct}dev.tsv',
                                  sep='\t',
                                  names=names)
    datasets['test'] = pd.read_csv(f'{folder_struct}test.tsv',
                                   sep='\t',
                                   names=names)

    if duplicates:
        for name in datasets.keys():
            datasets[name].drop_duplicates(inplace=True,
                                           subset=['Sentence (x)'])

    for name, data in datasets.items():
        if duplicates:
            file_writer(folder_struct, data, name, 'deduplicated')

        data_undersampled = sampling(data, replace=False,
                                     sampling_type='under')
        file_writer(folder_struct, data_undersampled, name, 'undersampled')

        data_oversampled = sampling(data, replace=True, sampling_type='over')
        file_writer(folder_struct, data_oversampled, name, 'oversampled')


if __name__ == '__main__':
    main()
