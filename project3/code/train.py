#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

import argparse
import os

from extract_features import main_train
from knn import store_classifier
from utils import FINAL_FEATURES, K_MEANS, DB_SCAN


def main():
    """
    Main method
    :return: None
    """
    parser = argparse.ArgumentParser(description='Train or test the model.')
    parser.add_argument('--kfold', dest='kfold', required=False, type=int, default=0,
                        help='1 for k fold cross validation, otherwise it generated model')
    args = parser.parse_args()
    train = False
    if args.kfold == 1:
        train = True
    else:
        if not os.path.exists(FINAL_FEATURES):
            main_train()
    # training is done.
    # Ground truth
    # ground_truth_bins = bin_all_patients()
    # ground_truth = bin_to_cls_count(ground_truth_bins)

    # K-means
    # kmean_gt_map, kmeans_res_cls, kmeans_array = kmeans_cluster_mapper()

    # DBscan
    # dbscan_gt_map, dbscan_res_cls, dbscan_array = db_scan_cluster_mapper()

    # Train KNN
    # kmeans
    print("\n K-mean clustering on training data:")
    store_classifier(25, K_MEANS, train)
    print("\n DBSCAN clustering on training data:")
    store_classifier(25, DB_SCAN, train)


if __name__ == '__main__':
    main()
