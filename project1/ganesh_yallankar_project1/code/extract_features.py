#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

import pandas as pd
import numpy as np
from sklearn import preprocessing
from tsfresh import extract_features
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

LIMIT = 30

series_df_all = pd.DataFrame()
thirty_list = np.arange(0, 30, 1).tolist()
all_ele_id = thirty_list * 33

# Iterate for all 5 patients
for x in range(1, 6):
    # generate file name
    data_path = "../data/CGMSeriesLunchPat{}.csv".format(x)
    ts_path = "../data/CGMDatenumLunchPat{}.csv".format(x)

    # read csv files
    data_cgm = pd.read_csv(data_path, usecols=np.arange(0, LIMIT, 1).tolist())
    ts_cgm = pd.read_csv(ts_path, usecols=np.arange(0, LIMIT, 1).tolist())

    ### Data preproccessing start
    # drop missing data and replace nans with mean
    nan_rows = data_cgm.apply(lambda y: y.notna().sum() > 27, axis=1)
    nan_indexes_list = nan_rows[nan_rows == False].index

    data_cgm.drop(nan_indexes_list, axis=0, inplace=True)
    data_cgm = data_cgm.bfill(axis=1).ffill(axis=1)
    data_cgm.fillna(axis=1, method='backfill', inplace=True)
    data_cgm.fillna(axis=1, method='ffill', inplace=True)

    ts_cgm.drop(nan_indexes_list, axis=0, inplace=True)
    ts_cgm.fillna(axis=1, method='backfill', inplace=True)

    ## put data in range of 0 - 1
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_data_cgm = min_max_scaler.fit_transform(data_cgm)

    ## now reverse array as data is in reverse timestamp order
    norm_rev_data_cgm = np.fliplr(normalized_data_cgm)
    rev_ts_cgm = np.fliplr(ts_cgm)
    ### Data Preprocessing complete

    ### make data linear for tsfresh
    ts_in_series = rev_ts_cgm.flatten()
    data_in_series = norm_rev_data_cgm.flatten()
    all_ele_id = thirty_list * int(len(ts_in_series) / 30)
    time_data_combined = np.vstack((all_ele_id, ts_in_series, data_in_series)).T
    cgm_t_d_df = pd.DataFrame(data=time_data_combined, columns=['_id', 'timestamp', 'cgmseries'])
    series_df_all = pd.concat([series_df_all, cgm_t_d_df], axis=0)

### Feature: Discrete Fourier transform
print("\nDiscrete Fourier Transform.\n")

fft_real_coefs = [{'coeff': 0, 'attr': 'real'}, {'coeff': 1, 'attr': 'real'}, {'coeff': 2, 'attr': 'real'},
                  {'coeff': 3, 'attr': 'real'}, {'coeff': 4, 'attr': 'real'}, {'coeff': 5, 'attr': 'real'}]
fft_features = extract_features(timeseries_container=series_df_all, column_id='_id', column_sort="timestamp",
                                column_value="cgmseries", default_fc_parameters=dict(fft_coefficient=fft_real_coefs))

fft_features_normalised = min_max_scaler.fit_transform(fft_features)
fft_features_normalised_df = pd.DataFrame(data=fft_features_normalised,
                                          columns=['fft_coefficient_0', 'fft_coefficient_1', 'fft_coefficient_2',
                                                   'fft_coefficient_3', 'fft_coefficient_4', 'fft_coefficient_5'])

### Feature FFT done.

### Feature Discrete Wavelet Transform
print("\nDiscrete Wavelet Transform.\n")
d_fc_params = {
    "cwt_coefficients": [{"widths": width, "coeff": coeff, "w": w} for width in [(1, 30)] for coeff in range(5) for w in
                         (1, 30)]}
dwt_features = extract_features(timeseries_container=series_df_all, column_id='_id', column_sort="timestamp",
                                column_value="cgmseries", default_fc_parameters=d_fc_params)

dwt_features_normalised = min_max_scaler.fit_transform(dwt_features)
dwt_features_normalised_df = pd.DataFrame(data=dwt_features_normalised,
                                          columns=['dwt_coef_0', 'dwt_coef_1', 'dwt_coef_2', 'dwt_coef_3', 'dwt_coef_4',
                                                   'dwt_coef_5', 'dwt_coef_6', 'dwt_coef_7', 'dwt_coef_8',
                                                   'dwt_coef_9'])

### Feature DWT done

### Power spectral density
print("\nPower spectral density.\n")
spkt_params = {"spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]]}
spkt_features = extract_features(timeseries_container=series_df_all, column_id='_id', column_sort="timestamp",
                                 column_value="cgmseries", default_fc_parameters=spkt_params)

spkt_features_normalised = min_max_scaler.fit_transform(spkt_features)
spkt_features_normalised_df = pd.DataFrame(data=spkt_features_normalised,
                                           columns=['spkt_coef_0', 'spkt_coef_1', 'spkt_coef_2'])

### psd done.

### Time series Features
print("\nTime series features.\n")
ts_params = dict(abs_energy=None, absolute_sum_of_changes=None, kurtosis=None, skewness=None, sample_entropy=None)
ts_features = extract_features(timeseries_container=series_df_all, column_id='_id', column_sort="timestamp",
                               column_value="cgmseries", default_fc_parameters=ts_params)

ts_features_normalised = min_max_scaler.fit_transform(ts_features)
ts_features_normalised_df = pd.DataFrame(data=ts_features_normalised,
                                         columns=['ts_coef_0', 'ts_coef_1', 'ts_coef_2', 'ts_coef_3', 'ts_coef_4'])

### TSF done

combined_df = pd.concat(
    [fft_features_normalised_df, dwt_features_normalised_df, spkt_features_normalised_df, ts_features_normalised_df],
    axis=1)

combined_df.to_csv('all_features.csv')

print("\nPrincipal component analysis.\n")
pca = PCA(n_components=5)
pca.fit(combined_df)
data = dict(Component1=pca.components_[0], Component2=pca.components_[1], Component3=pca.components_[2],
            Component4=pca.components_[3], Component5=pca.components_[4])

print("\nPCA fit done.\n")
plt.plot(data['Component1'])
plt.plot(data['Component2'])
plt.plot(data['Component3'])
plt.plot(data['Component4'])
plt.plot(data['Component5'])
plt.xlabel("Features")
plt.ylabel("Feature Variance for each component.")
plt.title('Principal component analysis')
plt.show()
pca_combined_df = pca.transform(combined_df)

## Get top 5 features
top_5_components = sorted(range(len(pca.components_[0])), key=lambda k: pca.components_[0][k])[0:5]
top_5_features = [combined_df.columns[i] for i in top_5_components]
print("Top features are:")
for feature in top_5_features:
    print(feature)

pca_df = pd.DataFrame.from_records(pca_combined_df, columns=top_5_features)
pca_df.to_csv('pca_all_features.csv')

for column in pca_df.columns.values:
    plt.scatter(range(len(pca_df)), pca_df[column])
    plt.title="PCA "+str(column)    
    plt.show()
## plot all feature graphs.

fft_features_normalised_df.plot(kind='bar', rot=0, lw=2, colormap='inferno', figsize=(10, 4),
                                title="Fast Fourier Transform", subplots=True, layout=(3, 2), legend=False)
dwt_features_normalised_df.plot(kind='line', rot=0, lw=2, colormap='inferno', figsize=(10, 4),
                                title="Discrete Wavelet Transform", subplots=True, layout=(3, 4), legend=False)
spkt_features_normalised_df.plot(kind='density', rot=0, lw=2, colormap='inferno', figsize=(10, 4),
                                 title="Power spectral density", subplots=True, layout=(2, 2), legend=False)
ts_features_normalised_df.plot(kind='bar', rot=0, lw=2, colormap='inferno', figsize=(10, 4),
                               title="Time Series Features", subplots=True, layout=(2, 3), legend=False)

plt.show()
