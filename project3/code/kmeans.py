#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

import operator
import warnings

warnings.simplefilter("ignore")

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils import read_feature_csv
from binning import bin_all_patients


def k_means():
    data = read_feature_csv()
    all_features = data
    # data.iloc[:, 1:-1]
    # k_means = KMeans(n_clusters=6, random_state=0).fit_predict(all_features)
    clustering = MiniBatchKMeans(n_clusters=6, random_state=0, n_init=15, max_iter=10000, batch_size=30).fit_predict(
        all_features)
    k_means_fit = MiniBatchKMeans(n_clusters=6, random_state=0, n_init=15, max_iter=10000, batch_size=30).fit(
        all_features)
    # n_clusters=6, n_init=2, max_iter=1000)
    clustering = clustering + 1
    print("-" * 100)
    print(clustering)
    print("-" * 100)
    sse = k_means_fit.inertia_
    # print(sse)
    return clustering, sse


# def kmeans_sse():
#     data = read_feature_csv()
#     sse = {}
#     for k in range(1, 10):
#         kmeans = MiniBatchKMeans(n_clusters=k, max_iter=1000, batch_size=50).fit(data)
#         print(kmeans.cluster_centers_)
#         data["clusters"] = kmeans.labels_
#         print(data["clusters"])
#         sse[k] = kmeans.inertia_
#         # Inertia: Sum of distances of samples to their closest cluster center
#
#     plt.figure()
#     plt.plot(list(sse.keys()), list(sse.values()))
#     plt.xlabel("Number of cluster")
#     plt.ylabel("SSE")
#     plt.show()


def bin_to_cls_count(binned_array=None):
    sse = 0
    if binned_array is None:
        binned_array, sse = k_means()
    counter_bin = {}
    binned_list = binned_array.tolist()
    for i in range(len(binned_list)):
        ele = binned_list[i]
        if ele in counter_bin.keys():
            counter_bin[ele].append(i)
        else:
            counter_bin[ele] = [i]
    # print(counter_bin)
    return counter_bin, sse


def get_metrics(y_test=None, y_pred=None, sse=0, class_error=0):
    if y_test is None:
        y_test = bin_all_patients()
    if y_pred is None:
        y_pred, sse = k_means()
    p_r_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("Calculated Metrics: ")
    print("Classification Error  = {0:.2f}%".format(class_error))
    print("SSE                   = {0:.3f}".format(sse))
    print("Accuracy              = {0:.3f}".format(accuracy_score(y_test, y_pred)))
    print("Precision             = {0:.3f}".format(p_r_f1[0]))
    print("Recall                = {0:.3f}".format(p_r_f1[1]))
    print("F1 Score              = {0:.3f}".format(p_r_f1[2]))


def kmeans_cluster_mapper():
    matching_count = 0
    dummy = read_feature_csv()
    res_cls_array = [None] * dummy.shape[0]
    # print("row count is: ", dummy.shape[0])
    resulatant_cluster = {}
    kmeans_bins, k_sse = bin_to_cls_count()
    ground_truth_bins = bin_all_patients()
    ground_truth, dummy_sse = bin_to_cls_count(ground_truth_bins)
    cluster_map = {}
    x, y = [], []
    # print("+" * 100)
    for ele in ground_truth:
        a = ground_truth[ele]
        res = {}
        for kmeans_cls in kmeans_bins:
            b = kmeans_bins[kmeans_cls]
            d = [value for value in a if value in b]
            res[kmeans_cls] = len(set(d))
            # print(kmeans_cls, d)
        max_key = max(res.items(), key=operator.itemgetter(1))[0]
        matching_count = matching_count + max(res.items(), key=operator.itemgetter(1))[1]
        # print("-" * 100)
        # print(ele, max_key)
        cluster_map[ele] = max_key
        resulatant_cluster[ele] = kmeans_bins[max_key]
        for ele2 in kmeans_bins[max_key]:
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
    class_error = (matching_count*100)/dummy.shape[0]
    get_metrics(ground_truth_bins, res_cls_array, k_sse, class_error)
    return cluster_map, resulatant_cluster, res_cls_array


if __name__ == '__main__':
    cls_map, res_cls, rs = kmeans_cluster_mapper()
    print(cls_map)
    print(res_cls)

"""
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  1  0  0
  0  1  0  0  0  1  0  0  2  0  2  0  2  0  2  0  2  0  2  0  2  0  2  0
  2 -1  2  2  2  2  2  2  2  2  2 -1  2  2  2  2  2  2  2 -1  2  2  2  2
  2  2  2 -1  2  2  2  2  2  2  2 -1  2  2  2  2  2  2  2 -1  2  2  2  2
  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
  2  2  2  2  2]
"""
