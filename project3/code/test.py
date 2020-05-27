#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

from knn import predict_knn
from utils import K_MEANS, DB_SCAN
import argparse
import sys
from extract_features import extract_all_features, PARENT_DIR

OUTPUT_CSV = PARENT_DIR + "/output.csv"


def main():
    """
    Main method for classification: Classifies test data into meal/no_meal.
    :return: None
    """
    parser = argparse.ArgumentParser(description='Classifies Data as Meal/NoMeal data')
    parser.add_argument('--file', dest='input_file', required=True,
                        help='pass the ABSOLUTE path to the csv file')

    args = parser.parse_args()
    input_file = args.input_file
    if '.csv' not in input_file:
        sys.exit('Please provide CSV input file')
    features = extract_all_features(csv_path=input_file)
    k_means_predicted_res = predict_knn(K_MEANS, features)
    k_means_predicted_res = k_means_predicted_res + 1
    db_scan_predicted_res = predict_knn(DB_SCAN, features)
    db_scan_predicted_res = db_scan_predicted_res + 1
    db_scan_predicted_res[db_scan_predicted_res == 0] = -1
    k_means_res = k_means_predicted_res.tolist()
    db_scan_res = db_scan_predicted_res.tolist()
    output_handle = open(OUTPUT_CSV, "w+")
    print("Output:\n")
    print("DBScan       K-means")
    for i in range(len(k_means_res)):
        print("   {}            {}".format(db_scan_res[i], k_means_res[i]))
        res_str = "{},{}\n".format(db_scan_res[i], k_means_res[i])
        output_handle.write(res_str)


if __name__ == '__main__':
    main()
