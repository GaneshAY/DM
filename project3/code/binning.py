#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

import json
import os
from pathlib import Path
import numpy as np
from utils import store_to_file, DATA_BINS

PWD = str(os.path.abspath(os.getcwd()))
PARENT_DIR = str(Path(str(os.path.abspath(os.getcwd()))).parent)
DATA_DIR = "{}/data".format(PARENT_DIR)
DROPPED_TUPLES = PARENT_DIR + "/data/dropped_data.json"
BINS = np.arange(-19, 120, 20)


def read_csv(file_pattern, write=False, access_mode="r"):
    if write:
        access_mode = "w+"
    file_name = "{}/{}.csv".format(PWD, file_pattern)
    try:
        file_handle = open(file_name, access_mode)
        return file_name, file_handle
    except IOError:
        print("File not accessible")
        exit(1)


def put_to_bin(subject=0, meal_amount_file=""):
    """
    Assumption:
        for each patient we'll add to same cluster with incremented numbering (50 * patient + index)
    """
    if subject in range(0, 5):
        if not meal_amount_file:
            meal_amount_file = "{}/mealAmountData{}.csv".format(DATA_DIR, subject + 1)
    with open(meal_amount_file, "r") as meal_file:
        meal_amount_list = meal_file.readlines()
    meal_amount_list = [int(y.strip()) for y in meal_amount_list]
    binned_data = np.digitize(meal_amount_list[:50], list(BINS))
    binned_data = [x - 1 for x in binned_data]
    binned_data = np.array(binned_data)
    dropped_data_file = open(DROPPED_TUPLES, "r")
    dropped_data = dropped_data_file.read()
    dropped_data_dict = json.loads(dropped_data)
    patient_dropped_data = dropped_data_dict[int(subject)]
    binned_data = np.delete(binned_data, patient_dropped_data[str(subject + 1)])
    return binned_data


def bin_all_patients():
    all_patient_clusters = []
    for patient in range(0, 5):
        all_patient_clusters.append(put_to_bin(patient))

    all_binned = np.concatenate(all_patient_clusters)
    store_to_file(all_binned, DATA_BINS)
    # print(all_binned)
    return all_binned


def bin_to_cls_count(binned_array=None):
    if binned_array is None:
        binned_array = bin_all_patients()
    counter_bin = {}
    binned_list = binned_array.tolist()
    for i in range(len(binned_list)):
        ele = binned_list[i]
        if ele in counter_bin.keys():
            counter_bin[ele].append(i)
        else:
            counter_bin[ele] = [i]
    print(counter_bin)
    return counter_bin


if __name__ == '__main__':
    bin_to_cls_count()

"""
[3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 5 2 4 2 4 3 5 2 5 0 4 1 5 0 4 1 5 0 4 1
 5 0 4 2 5 2 1 0 2 0 4 0 1 2 1 2 1 0 4 0 1 2 1 2 1 0 4 0 1 2 1 2 1 0 4 2 1
 2 0 4 0 1 2 1 2 1 0 4 0 0 0 0 3 0 3 0 3 0 2 3 0 3 3 0 0 2 3 0 3 0 2 3 0 0
 0 3 3 1 0 0 0 0 0 3 3 0 0 0 3 3 1 1 1 0 1 1 0 0 2 3 1 1 1 3 1 2 0 2 3 1 1
 1 3 1 2 0 2 3 1 1 1 3 1 2 0 2 3 1 1 1 3 1 2 0 2 3 1 1 3 3 0 3 3 3 0 3 3 3
 0 3 3 3 0 3 3 1 1 0 3 0 3 3 1 1 0 3 3 1 1 0 3 3 1 1 0 3 2 0 0 3 0 3 2 0]
 """
