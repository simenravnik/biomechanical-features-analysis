import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import statistics
from src import helper


def load_data():
    """
    Reading dataset
    :return: a pair of pandas data frames
    """
    data_2c = pd.read_csv('../data/dataset-2C.csv')
    data_3c = pd.read_csv('../data/dataset-3C.csv')

    return data_2c, data_3c


def plot_distribution(x):
    """
    Plotting distribution histogram on which is fitted kernel density estimate (KDE)
    """
    sns.distplot(x)
    plt.show()


def remove_outliers(df):
    """
    Remove values that are more than three std deviations away from the mean

    :param df: all data as pandas data frame
    """
    outliers = {}
    for col in df.columns:
        if str(df[col].dtype) != 'object':
            df = df[np.abs(df[col] - df[col].mean()) < (3 * df[col].std())]
            olrs = df[~(np.abs(df[col] - df[col].mean()) < (3 * df[col].std()))]
            outliers = pd.DataFrame(olrs)
    return df, outliers


def predict_cross_validation(X, y, model, num_splits=10, name=None):
    """
    Calculate k-fold cross validation on train data

    :param X: train data which will be used in k-fold cross validation
    :param y: train class data
    :param model: function that returns prediction model
    :param num_splits: number of splits of cross validation
    :param name: name of the prediction model
    :return: mean accuracy of k-fold cross validation
    """
    # storing all K-fold accuracies of k
    accuracies = []
    f1_score_arr = []

    # dictionary for storing sensitivities and specificities of Hernia and Spondylolisthesis
    diagnosis_dict = {
        "sensitivity_hernia_arr": [],
        "specificity_hernia_arr": [],
        "sensitivity_normal_arr": [],
        "specificity_normal_arr": [],
        "sensitivity_spondylolisthesis_arr": [],
        "specificity_spondylolisthesis_arr": []
    }

    # for calculating roc curve
    y_scores = None
    y_tests = None
    classes = None

    # splitting train data again into K sets to perform cross validation
    k_fold = StratifiedKFold(n_splits=num_splits, shuffle=True)
    for train_index, test_index in k_fold.split(X, y):
        # K-fold train and test data set
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # get wanted prediction classifier
        classifier = model(x_train, y_train)

        # fitting k-fold train data into the model
        classifier.fit(x_train, y_train)

        classes = classifier.classes_

        # accuracy of current split
        accuracy = classifier.score(x_test, y_test)
        # predictions
        y_pred = classifier.predict(x_test)

        # update predictions if num of classes is 2, because of different threshold
        if len(classifier.classes_) == 2:
            # predict probabilities to be class 1 (Normal)
            y_pred_prob = classifier.predict_proba(x_test)[:, 1]
            y_pred = predict_with_threshold(y_pred_prob, 0.7, classes)

            # add to global scores and test
            if y_scores is None:
                y_scores = y_pred_prob
                y_tests = y_test
            else:
                y_scores = np.concatenate((y_scores, y_pred_prob), axis=0)
                y_tests = np.concatenate((y_tests, y_test), axis=0)
        elif name == 'logistic_regression':
            # predict confidence scores for each class if model is logistic regression
            y_score = classifier.decision_function(x_test)

            if y_scores is None:
                y_scores = y_score
                y_tests = y_test
            else:
                y_scores = np.concatenate((y_scores, y_score), axis=0)
                y_tests = np.concatenate((y_tests, y_test), axis=0)

        f1 = f1_score(y_test, y_pred, average='macro')

        # calculating sensitivity and specificity of current result
        df = helper.get_classification_report(y_test, y_pred, classes)
        # dictionary for storing current sensitivity and specificity results
        current_diagnosis = {
            "sensitivity_hernia": df['Hernia']['Sensitivity'],
            "specificity_hernia": df['Hernia']['Specificity'],
            "sensitivity_normal": df['Normal']['Sensitivity'],
            "specificity_normal": df['Normal']['Specificity']
        }

        # appending calculated sensitivity and specificity to diagnosis dictionary
        diagnosis_dict["sensitivity_hernia_arr"].append(current_diagnosis["sensitivity_hernia"])
        diagnosis_dict["specificity_hernia_arr"].append(current_diagnosis["specificity_hernia"])
        diagnosis_dict["sensitivity_normal_arr"].append(current_diagnosis["sensitivity_normal"])
        diagnosis_dict["specificity_normal_arr"].append(current_diagnosis["specificity_normal"])

        # appending spondylolisthesis if we have 3 classes
        if len(classifier.classes_) == 3:
            current_diagnosis["sensitivity_spondylolisthesis"] = df['Spondylolisthesis']['Sensitivity']
            current_diagnosis["specificity_spondylolisthesis"] = df['Spondylolisthesis']['Specificity']
            diagnosis_dict["sensitivity_spondylolisthesis_arr"].append(
                current_diagnosis["sensitivity_spondylolisthesis"])
            diagnosis_dict["specificity_spondylolisthesis_arr"].append(
                current_diagnosis["specificity_spondylolisthesis"])

        # adding current split accuracy to accuracies of current k
        accuracies.append(accuracy)
        f1_score_arr.append(f1)

    # average numbers
    print("Average accuracy = ", np.mean(accuracies))
    print("F1 score (macro) = ", np.mean(f1_score_arr))

    # calculating confidence intervals for sensitivity and specificity
    diagnosis_conf_dict = helper.calculate_confidence_intervals_t(
        diagnosis_dict["sensitivity_hernia_arr"],
        diagnosis_dict["specificity_hernia_arr"],
        diagnosis_dict["sensitivity_normal_arr"],
        diagnosis_dict["specificity_normal_arr"],
        diagnosis_dict["sensitivity_spondylolisthesis_arr"],
        diagnosis_dict["specificity_spondylolisthesis_arr"]
    )

    print("\nHernia sensitivity = ", np.mean(diagnosis_dict["sensitivity_hernia_arr"]),
          "   (95%CI ", diagnosis_conf_dict["sensitivity_hernia_conf"], ")")
    print("Hernia specificity = ", np.mean(diagnosis_dict["specificity_hernia_arr"]),
          "   (95%CI ", diagnosis_conf_dict["specificity_hernia_conf"], ")")

    print("\nNormal sensitivity = ", np.mean(diagnosis_dict["sensitivity_normal_arr"]),
          "   (95%CI ", diagnosis_conf_dict["sensitivity_normal_conf"], ")")
    print("Normal specificity = ", np.mean(diagnosis_dict["specificity_normal_arr"]),
          "   (95%CI ", diagnosis_conf_dict["specificity_normal_conf"], ")")

    if len(classes) == 3:
        print("\nSpondylolisthesis sensitivity = ", np.mean(diagnosis_dict["sensitivity_spondylolisthesis_arr"]),
              "   (95%CI ", diagnosis_conf_dict["sensitivity_spondylolisthesis_conf"], ")")
        print("Spondylolisthesis specificity = ", np.mean(diagnosis_dict["specificity_spondylolisthesis_arr"]),
              "   (95%CI ", diagnosis_conf_dict["specificity_spondylolisthesis_conf"], ")")

    if len(classes) == 2:
        # plot roc if we have two classes
        helper.plot_roc(y_tests, y_scores)
    elif name == 'logistic_regression':
        # plot roc if model is logistic regression
        helper.plot_roc_multiclass(y_scores, y_tests, classes)


def predict_with_threshold(y_pred_prob, threshold, classes):
    y_pred = list(map(lambda x: classes[1] if (x > threshold) else classes[0], y_pred_prob))
    return y_pred


def knn(X, y):
    # calculating optimal k for KNN prediction model
    # optimal_k = helper.calculate_optimal_k(X, y)
    optimal_k = 8
    # print(optimal_k)
    return KNeighborsClassifier(n_neighbors=optimal_k)


def decision_tree(X, y):
    return DecisionTreeClassifier(max_depth=3)


def logistic_regression_classifier(X, y):
    return LogisticRegression(multi_class='ovr')


def random_forest_classifier(X, y):
    return RandomForestClassifier(n_estimators=128)


def neural_network(X, y):
    return MLPClassifier(solver='lbfgs', max_iter=10000)


if __name__ == '__main__':
    # load data
    data_2c, data_3c = load_data()

    # removing outliers from the dataset
    data, _ = remove_outliers(data_3c)

    # separating the features from the target value
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values

    # PREDICTION
    print("\nKNN:")
    predict_cross_validation(X, y, knn)                                 # knn
    print("\nDecision tree:")
    predict_cross_validation(X, y, decision_tree)                       # decision tree
    print("\nLogistic regression:")
    predict_cross_validation(X, y, logistic_regression_classifier, name="logistic_regression")  # logistic regression
    print("\nRandom forest:")
    predict_cross_validation(X, y, random_forest_classifier)            # random forest
    print("\nNeural network:")
    predict_cross_validation(X, y, neural_network)                      # neural network

    # PREDICTING WITHOUT SPONDYLOLISTHESIS
    # remove Spondylolisthesis from data to run algorithm again
    filtered_data = data[data['class'] != 'Spondylolisthesis']
    # separating the features from the target value
    filtered_X = filtered_data.iloc[:, 0:-1].values
    filtered_y = filtered_data.iloc[:, -1].values

    print("\nLogistic regression (without Spondylolisthesis):")
    predict_cross_validation(filtered_X, filtered_y, logistic_regression_classifier)
