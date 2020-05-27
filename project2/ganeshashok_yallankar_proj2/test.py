#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

## test.py

from extract_features import *
import argparse

PWD = str(os.path.abspath(os.getcwd()))


def load_classifier(file_name, file_path="models"):
    """
    load classifier model from file
    :param file_name: classifier name
    :param file_path: model directory
    :return: returns model object
    """
    file_dir = PWD + "/" + file_path
    file_path = file_dir + "/" + file_name
    return pickle.load(open(file_path, "rb"))


def classifier(input_file, classifier_name="classifier"):
    """
    Pre-process, and classify the given data as per model.
    :param classifier_name: pass classifier name(in case of multiple classifiers.
    :param input_file: custom input csv data file.
    :return: None, prints predicted as Meal(1)/Nomeal(0).
    """
    features = extract_all_features(csv_path=input_file)
    features = normalization(features, train=False)
    output_file = "output.csv"
    output = pd.DataFrame()
    clf = load_classifier(file_name='{0}.pkl'.format(classifier_name))
    output["Meal_NoMeal"] = clf.predict(features)
    output = output.astype(int)
    if os.path.exists(output_file):
        os.remove(output_file)
    output.to_csv(output_file, index=False)
    print("Convention: Meal - 1, NoMeal - 0.")
    print(output.to_string())
    print("Results stored in {}".format(output_file))


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

    classifier(input_file)


if __name__ == '__main__':
    main()
