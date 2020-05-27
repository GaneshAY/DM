#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

## train.py

from extract_features import *
import argparse
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PWD = str(os.path.abspath(os.getcwd()))


def read_feature_csv():
    """
    Reads given csv file into a pandas dataframe
    :return: Pandas dataframe containing raw data.
    """
    file_path = PWD + "/" + FINAL_FEATURES
    return pd.read_csv(file_path, dtype=np.float)


def combined_classifier(train=False):
    """
    Generate and train/test the classifier
    :param train: True/False
    :return: None, prints - accuracy, precision, recall and f1 score.
    """
    data = read_feature_csv()
    all_features = data.iloc[:, 1:-1]
    meal_no_meal_class = data.iloc[:, -1]
    models = [('RF', RandomForestClassifier()), ('ABC', AdaBoostClassifier(n_estimators=500, algorithm='SAMME')),
              ('ETC', ExtraTreesClassifier(n_estimators=500, max_depth=11, random_state=23))]
    classifier_model = VotingClassifier(estimators=models)

    if train:
        ## to train the classifier
        print("Training the classifier...")
        classifier_model.fit(all_features, meal_no_meal_class)
        store_to_file(classifier_model, 'classifier.pkl')
    else:
        ## K fold cross validation.
        x_train, x_test, y_train, y_test = train_test_split(all_features, meal_no_meal_class, test_size=0.25,
                                                            random_state=30)
        classifier_model.fit(x_train, y_train.values.ravel())
        y_pred = classifier_model.predict(x_test)
        p_r_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print("Calculated Metrics: ")
        print("Accuracy  = {0:.3f}".format(accuracy_score(y_test, y_pred)))
        print("Precision = {0:.3f}".format(p_r_f1[0]))
        print("Recall    = {0:.3f}".format(p_r_f1[1]))
        print("F1 Score  = {0:.3f}".format(p_r_f1[2]))


def store_to_file(data, file_name, dir_path="models"):
    """
    Stores the classifier as pickle dump.
    :param data: classifier object.
    :param file_name: classifier file name.
    :param dir_path: subdirectory for model.
    :return: None
    """
    file_dir = PWD + "/" + dir_path
    file_path = file_dir + "/" + file_name
    try:
        os.mkdir(file_dir)
    except:
        pass
    pickle.dump(data, open(file_path, 'wb'))
    print("Stored model into pickle: {}".format(file_path))


def main():
    """
    Main method
    :return: None
    """
    parser = argparse.ArgumentParser(description='Train or test the model.')
    parser.add_argument('--test', dest='test', required=False, type=int, default=0,
                        help='1 for test, otherwise it generated model')
    args = parser.parse_args()
    train = True
    if args.test == 1:
        train = False
    else:
        if not os.path.exists(FINAL_FEATURES):
            main_train()

    combined_classifier(train)


def kmeans():
    from sklearn.cluster import KMeans
    import numpy as np

    data = read_feature_csv()
    all_features = data.iloc[:, 1:-1]
    k_means = KMeans(n_clusters=8, random_state=0).fit_predict(all_features)
    print("-" * 100)
    print(k_means)
    print("-" * 100)

# kmeans()

def dbscan():
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    data = read_feature_csv()
    all_features = data.iloc[:, 1:-1]
    #clustering = DBSCAN().fit(all_features)
    print("-" * 100)
    #print(clustering.labels_)
    print("-" * 100)
    X = StandardScaler().fit_transform(all_features)
    clustering = DBSCAN(eps=1, min_samples=2).fit_predict(X)
    print("-" * 100)
    print(clustering)
    print("-" * 100)



dbscan()

# if __name__ == '__main__':
#    main()
