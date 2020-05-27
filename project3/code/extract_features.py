#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features

from utils import PCA_COMPONENTS

PWD = str(os.path.dirname(os.path.realpath(__file__)))
PARENT_DIR = str(Path(str(os.path.dirname(os.path.realpath(__file__)))).parent)
FINAL_FEATURES = PARENT_DIR + "/features/final_features.csv"
DROPPED_TUPLES = PARENT_DIR + "/data/dropped_data.json"


def fill_csv(file_path, count=30):
    """
    Add additional commas to csv as some data are missing (max missing can be 30).
    :param file_path: input data file path (csv format).
    :param count: number of data elements per row.
    :return: Nothing, exits if method fails.
    """
    print("Input file: " + file_path)
    if not os.path.exists(file_path):
        print("Error: File {} does not exist.".format(file_path))
        sys.exit(1)
    additional_data = "," * count
    try:
        with open(file_path, 'r') as f:
            file_lines = [''.join([x.strip(), additional_data, '\n']) for x in f.readlines()]

        with open(file_path, 'w') as f:
            f.writelines(file_lines)

    except Exception as e:
        print("Exception occurred while reading file! Exiting...")
        print(e)
        sys.exit(1)


def fft_feature(tsfresh_data_format):
    """
    Feature1: Fast fourier transform.
    :param tsfresh_data_format: cgm data is np-nd_array format.
    :return: fft features.
    """
    params = dict(fft_coefficient=[dict(coeff=0, attr='real'), dict(coeff=1, attr='real'),
                                   dict(coeff=2, attr='real'), dict(coeff=3, attr='real'),
                                   dict(coeff=4, attr='real')])
    fast_fourier_transform = extract_features(timeseries_container=tsfresh_data_format, column_id='id',
                                              column_sort="ts", default_fc_parameters=params)
    return fast_fourier_transform


def dwt_feature(data):
    """
    Feature2: This method returns Discrete wavelet transform values from input data.
    :param data: Pandas dataframe for cgm data.
    :return: Pandas dataframe with feature values obtained from DWT.
    """

    data_frame = pd.DataFrame(
        columns=['dwt_ca_coeff0', 'dwt_ca_coeff1', 'dwt_ca_coeff2', 'dwt_ca_coeff3', 'dwt_ca_coeff4', 'dwt_cd_coeff0',
                 'dwt_cd_coeff1', 'dwt_cd_coeff2', 'dwt_cd_coeff3', 'dwt_cd_coeff4'])
    for idx, row in data.iterrows():
        (coefficient_a, coefficient_d) = pywt.dwt(row, wavelet='sym4', mode='symmetric')
        data_frame.loc[idx] = np.concatenate((coefficient_a[:5], coefficient_d[:5]))
    return data_frame


def feature_three(tsfresh_data_format):
    """
    Feature3: Kurtosis, Skewness, mean, median, variance, entropy, min and max.
    :param tsfresh_data_format: cgm data is np-nd_array format.
    :return: Pandas dataframe with feature values obtained from FFT.
    """

    params = dict(abs_energy=None, absolute_sum_of_changes=None, kurtosis=None, skewness=None,
                  sample_entropy=None, linear_trend=[{"attr": "slope"}], variance=None, maximum=None, mean=None,
                  median=None, minimum=None)
    feature = extract_features(timeseries_container=tsfresh_data_format, column_id='id', column_sort="ts",
                               default_fc_parameters=params)
    return feature


def feature_four(data):
    """
    Feature4: High and low mean.
    :param data: Pandas dataframe for cgm data.
    :return: Pandas dataframe with feature values obtained from high and low mean.
    """

    feature = pd.DataFrame()
    feature['cgm_mean_h'] = data['cgm__maximum'] - data['cgm__mean']
    feature['cgm_mean_l'] = data['cgm__maximum'] - data['cgm__mean']
    return feature


def csv_to_data_frame(patient=None, meal=True, csv_path=""):
    """
    This method converts csv into pandas dataframe and np-nd_array format. Involves data cleaning and extrapolation.
    :param patient: Number of patient(1..5)
    :param meal: True/False if meal based data or not.
    :param csv_path: If custom csv file is passed as input. filepath to the file.
    :return: Pandas dataframe and np-nd_array for data, timestamp.
    """

    data_path = PARENT_DIR + '/data'
    if csv_path:
        fill_csv(csv_path)
        data = pd.read_csv(csv_path, header=None, dtype=np.float, usecols=np.arange(30))
    else:
        if meal:
            file_path = data_path + '/mealData{}.csv'.format(patient)
        else:
            file_path = data_path + '/Nomeal{}.csv'.format(patient)

        fill_csv(file_path)
        data = pd.read_csv(file_path, header=None, skiprows=0, dtype=np.float,
                           usecols=np.arange(30), nrows=50)

    time_tuples = []
    for x in range(0, len(data)):
        time_tuples.append(list(range(0, 30)))
    timestamp = pd.DataFrame(time_tuples)

    # Reverse data as it is in reverse order.
    timestamp = timestamp[timestamp.columns.tolist()[::]]
    data = data[data.columns.tolist()[::-1]]

    # data cleanup
    # Remove data tuples if values are missing for more than 5 fields.
    # Else, extrapolate data to missing values.
    tuples = data.apply(lambda g: g.notna().sum() > 25, axis=1)
    dropped_tuples = {patient: list(tuples[tuples == False].index)}
    # print("Patient is {} and dropped tuples are: {}.".format(patient, dropped_tuples))
    data.drop(tuples[tuples == False].index, axis=0, inplace=True)
    data.fillna(axis=1, method='ffill', inplace=True)
    data.fillna(axis=1, method='backfill', inplace=True)

    # timestamp cleanup
    timestamp.drop(tuples[tuples == False].index, axis=0, inplace=True)
    timestamp.fillna(axis=1, method='backfill', inplace=True)

    # Converts data in np array format for tsfresh module
    l_idx = data.index[0]
    h_idx = data.index[-1]
    tsfresh_data_format = pd.concat([timestamp.T[l_idx].rename('ts'), data.T[l_idx].rename('cgm')], axis=1)
    tsfresh_data_format['id'] = l_idx

    for i in range(l_idx + 1, h_idx + 1):
        if i not in data.index:
            continue
        ts_data_frame = pd.concat([timestamp.T[i].rename('ts'), data.T[i].rename('cgm')], axis=1)
        ts_data_frame['id'] = i
        tsfresh_data_format = tsfresh_data_format.append(ts_data_frame)
    return timestamp, data, tsfresh_data_format, dropped_tuples


def extract_all_features(patient=None, meal=True, csv_path=""):
    """
    Extract features for each patient.
    :param patient: Patient number between 1 to 5.
    :param meal: If meal data or not - True/False.
    :param csv_path: If custom data is passed - file path.
    :return: All the extracted features as pandas dataframe.
    """

    timestamp, data, tsfresh_data_format, dropped_tuples = csv_to_data_frame(patient, meal, csv_path)

    # Feature1: Fast fourier transform.
    fast_fourier_transform = fft_feature(tsfresh_data_format)

    # Feature2: Discrete Wavelet Transform
    discrete_wavelet_transform = dwt_feature(data)

    # Feature3: Kurtosis, Skewness, mean, median, variance, entropy, min and max.
    feature_3 = feature_three(tsfresh_data_format)

    # Feature4: Upper and lower mean.
    feature_4 = feature_four(feature_3)

    all_features = pd.concat([discrete_wavelet_transform, fast_fourier_transform, feature_3, feature_4], axis=1)
    all_features = all_features.replace([np.inf, -np.inf], 0)
    return all_features.rename(columns=lambda x: re.sub('"', '_', x)), dropped_tuples


def extract_features_all():
    """
    Extract features for all patient data and combine them into one feature file.
    :return: Stores all features into features.csv file.
    """
    all_features_df = pd.DataFrame()
    meal_amt = []
    all_dropped = []
    for meal in [True]:
        patient_features_df = pd.DataFrame()
        for patient in range(1, 6):
            data_path = PARENT_DIR + '/data'
            file_path = data_path + '/mealAmountData{}.csv'.format(patient)
            with open(file_path) as f:
                lines = f.read().splitlines()
            ml_p = lines[:50]
            meal_amt = meal_amt + ml_p
            p_df, g = extract_all_features(patient, meal)
            all_dropped.append(g)
            patient_features_df = pd.concat([patient_features_df, p_df], axis=0)
        all_features_df = pd.concat([all_features_df, patient_features_df], axis=0)

    pca = PCA(n_components=PCA_COMPONENTS).fit(all_features_df)

    top_components = sorted(range(len(pca.components_[0])), key=lambda k: pca.components_[0][k])[0:PCA_COMPONENTS]
    top_features = [all_features_df.columns[i] for i in top_components]
    pca_2d = pca.transform(all_features_df)
    pca_2d = StandardScaler().fit_transform(pca_2d)
    if os.path.exists(FINAL_FEATURES):
        os.remove(FINAL_FEATURES)
    pca_pd = pd.DataFrame(data=pca_2d, index=[i for i in range(len(pca_2d))], columns=top_features)
    pca_pd.to_csv(FINAL_FEATURES)
    if os.path.exists(DROPPED_TUPLES):
        os.remove(DROPPED_TUPLES)
    with open(DROPPED_TUPLES, 'w') as f:
        json.dump(all_dropped, f)


def main_train():
    """
    Extract all features from training data.
    :return: None
    """
    extract_features_all()


def main():
    main_train()


if __name__ == '__main__':
    main()
