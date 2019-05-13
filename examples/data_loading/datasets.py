import os
import urllib
import os.path

import numpy as np
from pandas import read_csv

dirname = os.path.dirname(__file__)


def read_dataset(name, label=None, sensitive_attribute=None, fold=None):
    if name == 'crimes':
        y_name = label if label is not None else 'ViolentCrimesPerPop'
        z_name = sensitive_attribute if sensitive_attribute is not None else 'racepctblack'
        fold_id = fold if fold is not None else 1
        return read_crimes(label=y_name, sensitive_attribute=z_name, fold=fold_id)
    else:
        raise NotImplemented('Dataset {} does not exists'.format(name))


def read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1):
    if not os.path.isfile('communities.data'):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", "communities.data")
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            "communities.names")

    # create names
    names = []
    with open('communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = read_csv('communities.data', names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)

    folds = data['fold'].astype(np.int)

    y = data[label].values
    to_drop += [label]

    z = data[sensitive_attribute].values
    to_drop += [sensitive_attribute]

    data.drop(to_drop + [label], axis=1, inplace=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    x = np.array(data.values)
    return x[folds != fold], y[folds != fold], z[folds != fold], x[folds == fold], y[folds == fold], z[folds == fold]
