import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as st


def cross_validation_score(x_train, y_train, classifier, num_splits=4):
    """
    Calculate k-fold cross validation on train data

    :param x_train: train data which will be used in k-fold cross validation
    :param y_train: train class data
    :param classifier: prediction model
    :param num_splits: number of splits of cross validation
    :return: mean accuracy of k-fold cross validation
    """
    # storing all K-fold accuracies of k
    accuracies = []
    # splitting train data again into K sets to perform cross validation
    k_fold = StratifiedKFold(n_splits=num_splits, shuffle=True)
    for train_index, test_index in k_fold.split(x_train, y_train):
        # K-fold train and test data set
        kf_x_train, kf_x_test = x_train[train_index], x_train[test_index]
        kf_y_train, kf_y_test = y_train[train_index], y_train[test_index]

        # fitting k-fold train data into the model
        classifier.fit(kf_x_train, kf_y_train)

        # accuracy of current split
        accuracy = classifier.score(kf_x_test, kf_y_test)

        # adding current split accuracy to accuracies of current k
        accuracies.append(accuracy)

    # our accuracy is the mean value of k-fold scores
    return np.mean(accuracies)


def calculate_optimal_k(x_train, y_train):
    """
    Calculating optimal K for KNN prediction model using k-fold cross validation

    :param x_train: train data
    :param y_train: train target
    :return: best k for KNN algorithm
    """
    # to find the optimal K for KNN algorithm we perform K-fold cross validation for each K and take the mean value.
    all_accuracies = []
    all_train_accuracies = []

    for k in range(2, 51):

        # storing all K-fold accuracies of k
        k_accuracies = []
        train_accuracies = []

        # splitting train data again into K sets to perform cross validation
        k_fold = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in k_fold.split(x_train, y_train):
            # defining prediction model with k nearest neighbours
            knn = KNeighborsClassifier(n_neighbors=k)

            # K-fold train and test data set
            kf_x_train, kf_x_test = x_train[train_index], x_train[test_index]
            kf_y_train, kf_y_test = y_train[train_index], y_train[test_index]

            # fitting k-fold train data into the model
            knn.fit(kf_x_train, kf_y_train)

            # accuracy of current split
            accuracy = knn.score(kf_x_test, kf_y_test)
            train_accuracy = knn.score(kf_x_train, kf_y_train)  # train accuracy

            # adding current split accuracy to accuracies of current k
            k_accuracies.append(accuracy)
            train_accuracies.append(train_accuracy)

        # appending mean of current k accuracies to all accuracies
        all_accuracies.append(np.mean(k_accuracies))
        all_train_accuracies.append(np.mean(train_accuracies))

    # plotting knn accuracies depending on k chart
    plot_knn_chart(all_accuracies, all_train_accuracies)

    # transforming list into numpy array
    all_accuracies = np.array(all_accuracies)

    # the best k for KNN algorithm is the index of max mean accuracies
    optimal_k = np.argmax(all_accuracies) + 1  # +1 because we dont start with zero

    return optimal_k


def plot_knn_chart(all_accuracies, all_train_accuracies):
    # plotting accuracy depending on k number
    plt.figure(figsize=[16, 9])
    plt.plot(np.arange(1, len(all_accuracies) + 1), all_accuracies, label="Testing accuracy")
    plt.plot(np.arange(1, len(all_train_accuracies) + 1), all_train_accuracies, label="Training accuracy")
    plt.xticks(np.arange(1, len(all_accuracies) + 1))
    plt.legend()
    plt.title('KNN accuracy depending on number of neighbours')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    # plt.savefig('../img/generated/knn.png', format='png', bbox_inches='tight')
    plt.show()


def plot_roc(y_test, y_pred_prob):
    """
    Plot ROC curve and calculate AUC

    :param y_pred_prob: probability to be class 1
    :param y_test: testing target array
    """

    tpr, fpr, threshold = roc_curve(y_test, y_pred_prob, pos_label='Hernia')

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'.format(auc(fpr, tpr)))
    # plt.plot(fpr[11], tpr[11], 'co', label="Threshold 0.5")
    # plt.plot(fpr[26], tpr[26], 'mo', label="Threshold 0.7")
    plt.xlabel('FPR (1 - specificity)')
    plt.ylabel('TPR (sensitivity)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_multiclass(y_score, y_test, classes):
    """
    Plot multiclass ROC curve and calculate AUC for each class

    :param y_score: predicted data
    :param y_test: testing target array
    :param classifier: logistic regression classifier
    :param classes: unique classes of target array
    """

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # binary representation of y_test for calculating micro average
    binary_y_test = []

    for idx, i in enumerate(classes):
        # array of 0 and 1 when 1 represents positions where value is equal to current (i-th) target
        target_arr = [1 if x == i else 0 for x in y_test]

        # calculating roc and auc
        fpr[idx], tpr[idx], _ = roc_curve(target_arr, y_score[:, idx])
        roc_auc[idx] = auc(fpr[idx], tpr[idx])

        # appending binary target array
        binary_y_test.append(target_arr)

    binary_y_test = np.array(binary_y_test)  # convert to numpy array

    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_y_test.ravel(order='F'), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2)

    for idx, i in enumerate(classes):
        plt.plot(fpr[idx], tpr[idx], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[idx]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR (1 - specificity)')
    plt.ylabel('TPR (sensitivity)')
    plt.legend(loc="lower right")
    plt.show()


def get_classification_report(y_test, y_pred, classes):
    """
    Print classification report for the classifiers prediction
    :param y_test: real y values
    :param y_pred: predicted y values
    :param classes: classification classes
    """

    # printing confusion matrix
    confusion = confusion_matrix(y_test, y_pred, labels=classes)

    sensitivity = recall_score(y_test, y_pred, labels=classes, average=None)
    specificity = specificity_score(confusion, classes)

    df = pd.DataFrame([sensitivity, specificity], index=['Sensitivity', 'Specificity'], columns=classes)

    return df


def specificity_score(confusion, classes):
    """
    Function for calculating sensitivity from confusion matrix
    :param confusion: confusion matrix (2D array)
    :param classes: array of unique class variables
    :return:
    """
    specificity = []
    for i in range(len(classes)):
        N = 0
        TN = 0
        for j in range(len(classes)):
            if j != i:
                for k in range(len(classes)):
                    N += confusion[j][k]

                    if k != i:
                        TN += confusion[j][k]

        specificity.append(TN / N)
    return specificity


def calculate_confidence_intervals(sensitivity_hernia_arr, specificity_hernia_arr, sensitivity_spondylolisthesis_arr,
                                   specificity_spondylolisthesis_arr):
    """
    function for calculating confidence intervals for sensitivity and specificity of hernia and spondylolisthesis
    :param sensitivity_hernia_arr: Hernia measurements sensitivity array
    :param specificity_hernia_arr: Hernia measurements specificity array
    :param sensitivity_spondylolisthesis_arr: Spondylolisthesis measurements sensitivity array
    :param specificity_spondylolisthesis_arr: Spondylolisthesis measurements specificity array
    :return:
    """
    # dictionary for storing confidence intervals of each value
    diagnosis_conf_dict = {}

    # calculating confidence intervals

    diagnosis_conf_dict["sensitivity_hernia_conf"] = st.norm.interval(alpha=0.95,
                                               loc=np.mean(sensitivity_hernia_arr),
                                               scale=st.sem(sensitivity_hernia_arr))
    diagnosis_conf_dict["specificity_hernia_conf"] = st.norm.interval(alpha=0.95,
                                               loc=np.mean(specificity_hernia_arr),
                                               scale=st.sem(specificity_hernia_arr))

    diagnosis_conf_dict["sensitivity_spondylolisthesis_conf"] = st.norm.interval(alpha=0.95,
                                               loc=np.mean(sensitivity_spondylolisthesis_arr),
                                               scale=st.sem(sensitivity_spondylolisthesis_arr))
    diagnosis_conf_dict["specificity_spondylolisthesis_conf"] = st.norm.interval(alpha=0.95,
                                               loc=np.mean(specificity_spondylolisthesis_arr),
                                               scale=st.sem(specificity_spondylolisthesis_arr))

    return diagnosis_conf_dict


def calculate_confidence_intervals_t(sensitivity_hernia_arr,
                                   specificity_hernia_arr,
                                   sensitivity_normal_arr,
                                   specificity_normal_arr,
                                   sensitivity_spondylolisthesis_arr,
                                   specificity_spondylolisthesis_arr):
    """
    function for calculating confidence intervals for sensitivity and specificity of hernia and spondylolisthesis
    :param sensitivity_hernia_arr: Hernia measurements sensitivity array
    :param specificity_hernia_arr: Hernia measurements specificity array
    :param specificity_normal_arr: Normal measurements sensitivity array
    :param sensitivity_normal_arr: Normal measurements specificity array
    :param sensitivity_spondylolisthesis_arr: Spondylolisthesis measurements sensitivity array
    :param specificity_spondylolisthesis_arr: Spondylolisthesis measurements specificity array
    :return:
    """
    # dictionary for storing confidence intervals of each value
    diagnosis_conf_dict = {}

    # calculating confidence intervals

    diagnosis_conf_dict["sensitivity_hernia_conf"] = st.t.interval(alpha=0.95, df=len(sensitivity_hernia_arr) - 1,
                                                                   loc=np.mean(sensitivity_hernia_arr),
                                                                   scale=st.sem(sensitivity_hernia_arr))
    diagnosis_conf_dict["specificity_hernia_conf"] = st.t.interval(alpha=0.95, df=len(specificity_hernia_arr) - 1,
                                                                   loc=np.mean(specificity_hernia_arr),
                                                                   scale=st.sem(specificity_hernia_arr))

    diagnosis_conf_dict["sensitivity_normal_conf"] = st.t.interval(alpha=0.95, df=len(sensitivity_normal_arr) - 1,
                                                                   loc=np.mean(sensitivity_normal_arr),
                                                                   scale=st.sem(sensitivity_normal_arr))
    diagnosis_conf_dict["specificity_normal_conf"] = st.t.interval(alpha=0.95, df=len(specificity_normal_arr) - 1,
                                                                   loc=np.mean(specificity_normal_arr),
                                                                   scale=st.sem(specificity_normal_arr))

    if len(sensitivity_spondylolisthesis_arr) != 0:
        diagnosis_conf_dict["sensitivity_spondylolisthesis_conf"] = st.t.interval(alpha=0.95, df=len(
            sensitivity_spondylolisthesis_arr) - 1,
                                                                                  loc=np.mean(
                                                                                      sensitivity_spondylolisthesis_arr),
                                                                                  scale=st.sem(
                                                                                      sensitivity_spondylolisthesis_arr))
        diagnosis_conf_dict["specificity_spondylolisthesis_conf"] = st.t.interval(alpha=0.95, df=len(
            specificity_spondylolisthesis_arr) - 1,
                                                                                  loc=np.mean(
                                                                                      specificity_spondylolisthesis_arr),
                                                                                  scale=st.sem(
                                                                                      specificity_spondylolisthesis_arr))

    return diagnosis_conf_dict
