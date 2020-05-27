#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

import pandas as pd
import os
from pathlib import Path
import numpy as np
import pickle

PWD = str(os.path.dirname(os.path.realpath(__file__)))
PARENT_DIR = str(Path(str(os.path.dirname(os.path.realpath(__file__)))).parent)
FINAL_FEATURES = PARENT_DIR + "/features/final_features.csv"
DROPPED_TUPLES = PARENT_DIR + "/data/dropped_data.json"
DATA_BINS = "data_bins"
PCA_COMPONENTS = 3
K_MEANS = "kmeans"
DB_SCAN = "dbscan"


def read_feature_csv():
    """
    Reads given csv file into a pandas dataframe
    :return: Pandas dataframe containing raw data.
    """
    return pd.read_csv(FINAL_FEATURES, dtype=np.float)


def store_to_file(data, file_name, dir_path="models"):
    """
    Stores the classifier as pickle dump.
    :param data: classifier object.
    :param file_name: classifier file name.
    :param dir_path: subdirectory for model.
    :return: None
    """
    file_dir = PARENT_DIR + "/" + dir_path
    file_path = file_dir + "/" + file_name
    try:
        # print(file_dir)
        os.mkdir(file_dir)
    except:
        pass
    pickle.dump(data, open(file_path, 'wb'))
    # print("Stored model into pickle: {}".format(file_path))


def load_file(file_name, dir_path="models"):
    file_dir = PARENT_DIR + "/" + dir_path
    file_path = file_dir + "/" + file_name
    pickle_data = pickle.load(open(file_path, 'rb'))
    return pickle_data
