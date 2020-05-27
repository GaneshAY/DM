#!/usr/bin/env python

__author__ = 'Ganesh Ashok Yallankar'
__email__ = 'gyallank@asu.edu'

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils import *
from kmeans import kmeans_cluster_mapper
from dbscan import db_scan_cluster_mapper
from binning import bin_all_patients


def store_classifier(num_neighbors=25, classifier_name=K_MEANS, train=False):
    if classifier_name == K_MEANS:
        cls_map, res_cls, rs_array = kmeans_cluster_mapper()
    else:
        cls_map, res_cls, rs_array = db_scan_cluster_mapper()

    all_features = read_feature_csv()
    all_features = all_features.iloc[:, 1:]
    classifier = KNeighborsClassifier(n_neighbors=num_neighbors)
    classifier_model = classifier.fit(all_features, rs_array)
    store_to_file(classifier_model, classifier_name+"_knn.pkl")
    if train:
        # K fold cross validation.
        ground_truth_bins = bin_all_patients()
        x_train, x_test, y_train, y_test = train_test_split(all_features, ground_truth_bins.tolist(), test_size=0.25, random_state=30)
        classifier.fit(x_train, y_train)
        y_pred = classifier_model.predict(x_test)
        p_r_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print("\n Calculated Metrics from K fold cross validation: ")
        print("Accuracy  = {0:.3f}".format(accuracy_score(y_test, y_pred)))
        print("Precision = {0:.3f}".format(p_r_f1[0]))
        print("Recall    = {0:.3f}".format(p_r_f1[1]))
        print("F1 Score  = {0:.3f}".format(p_r_f1[2]))
    return


def load_classifier(file_name="knn", file_path="models"):
    """
    load classifier model from file
    :param file_name: classifier name
    :param file_path: model directory
    :return: returns model object
    """
    file_dir = PARENT_DIR + "/" + file_path
    file_path = file_dir + "/" + file_name
    return pickle.load(open(file_path, "rb"))


def predict_knn(classifier_name, extracted):
    features, dropped = extracted
    pca = PCA(n_components=PCA_COMPONENTS).fit(features)
    top_components = sorted(range(len(pca.components_[0])), key=lambda k: pca.components_[0][k])[0:PCA_COMPONENTS]
    top_features = [features.columns[i] for i in top_components]
    pca_2d = pca.transform(features)
    pca_2d = StandardScaler().fit_transform(pca_2d)
    pca_pd = pd.DataFrame(data=pca_2d, index=[i for i in range(len(pca_2d))], columns=top_features)
    scaled_features = StandardScaler().fit_transform(pca_pd)
    classifier_model = load_classifier(file_name='{0}_knn.pkl'.format(classifier_name))
    res = classifier_model.predict(scaled_features)
    return res


if __name__ == '__main__':
    store_classifier(25, "kmeans-knn", False)
    predicted_res = predict_knn('/home/ganesh/Documents/Documents/DM/proj3_tf/data/mealData3.csv', "kmeans-knn")
    print(predicted_res)


