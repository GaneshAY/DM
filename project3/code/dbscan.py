#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

import warnings

warnings.simplefilter("ignore")

import operator

from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils import read_feature_csv
from binning import bin_all_patients


def db_scan():
    data = read_feature_csv()
    all_features = data
    clustering = DBSCAN(eps=4.5, min_samples=8).fit_predict(all_features)
    clustering = clustering + 1
    clustering[clustering == 0] = -1
    print("-" * 100)
    print(clustering)
    print("-" * 100)
    return clustering


def get_metrics(y_test=None, y_pred=None, class_error=0):
    if y_test is None:
        y_test = bin_all_patients()
    if y_pred is None:
        y_pred = db_scan()
    p_r_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("Calculated Metrics: ")
    print("Classification Error  = {0:.2f}%".format(class_error))
    # print("SSE                   = {0:.3f}".format(sse))
    print("Accuracy              = {0:.3f}".format(accuracy_score(y_test, y_pred)))
    print("Precision             = {0:.3f}".format(p_r_f1[0]))
    print("Recall                = {0:.3f}".format(p_r_f1[1]))
    print("F1 Score              = {0:.3f}".format(p_r_f1[2]))


def bin_to_cls_count(binned_array=None):
    if binned_array is None:
        binned_array = db_scan()
    counter_bin = {}
    binned_list = binned_array.tolist()
    for i in range(len(binned_list)):
        ele = binned_list[i]
        if ele in counter_bin.keys():
            counter_bin[ele].append(i)
        else:
            counter_bin[ele] = [i]
    # print(counter_bin)
    return counter_bin


def db_scan_cluster_mapper():
    matching_count = 0
    dummy = read_feature_csv()
    res_cls_array = [None] * dummy.shape[0]
    # print("row count is: ", dummy.shape[0])
    resultant_cluster = {}
    db_scan_bins = bin_to_cls_count()
    ground_truth_bins = bin_all_patients()
    ground_truth = bin_to_cls_count(ground_truth_bins)
    cluster_map = {}
    x, y = [], []
    # print("+" * 100)
    for ele in ground_truth:
        a = ground_truth[ele]
        res = {}
        for db_scan_cls in db_scan_bins:
            b = db_scan_bins[db_scan_cls]
            d = [value for value in a if value in b]
            res[db_scan_cls] = len(d)
            # print(db_scan_cls, d)
        max_key = max(res.items(), key=operator.itemgetter(1))[0]
        matching_count = matching_count + max(res.items(), key=operator.itemgetter(1))[1]
        # print("-" * 100)
        # print(ele, max_key)
        cluster_map[ele] = max_key
        resultant_cluster[ele] = db_scan_bins[max_key]
        for ele2 in db_scan_bins[max_key]:
            res_cls_array[ele2] = ele
        x.append(ele)
        y.append(max_key)
        # print("*" * 100)
    ms_cls = list(set(x) - set(y))[0]
    # print("missing cls is ",ms_cls)
    for x in range(len(res_cls_array)):
        if res_cls_array[x] is None:
            res_cls_array[x] = ms_cls
    # print(res_cls_array)
    class_error = (matching_count * 100) / dummy.shape[0]
    get_metrics(ground_truth_bins, res_cls_array, class_error)
    return cluster_map, resultant_cluster, res_cls_array


if __name__ == '__main__':
    db_scan()
    # cls_map, res_cls, rs = db_scan_cluster_mapper()
    # print(cls_map)
    # print(res_cls)

"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
"""
