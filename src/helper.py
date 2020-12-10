import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


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
    k_fold = KFold(n_splits=num_splits)
    for train_index, test_index in k_fold.split(x_train):
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

    for k in range(1, 51):

        # defining prediction model with k nearest neighbours
        knn = KNeighborsClassifier(n_neighbors=k)

        # storing all K-fold accuracies of k
        k_accuracies = []
        train_accuracies = []

        # splitting train data again into K sets to perform cross validation
        k_fold = KFold(n_splits=4)
        for train_index, test_index in k_fold.split(x_train):
            # K-fold train and test data set
            kf_x_train, kf_x_test = x_train[train_index], x_train[test_index]
            kf_y_train, kf_y_test = y_train[train_index], y_train[test_index]

            # fitting k-fold train data into the model
            knn.fit(kf_x_train, kf_y_train)

            # accuracy of current split
            accuracy = knn.score(kf_x_test, kf_y_test)
            train_accuracy = knn.score(kf_x_train, kf_y_train)    # train accuracy

            # adding current split accuracy to accuracies of current k
            k_accuracies.append(accuracy)
            train_accuracies.append(train_accuracy)

        # appending mean of current k accuracies to all accuracies
        all_accuracies.append(np.mean(k_accuracies))
        all_train_accuracies.append(np.mean(train_accuracies))

    # plotting knn accuracies depending on k chart
    # plot_knn_chart(all_accuracies, all_train_accuracies)

    # transforming list into numpy array
    all_accuracies = np.array(all_accuracies)

    # the best k for KNN algorithm is the index of max mean accuracies
    optimal_k = np.argmax(all_accuracies) + 1      # +1 because we dont start with zero

    return optimal_k


def plot_knn_chart(all_accuracies, all_train_accuracies):
    # plotting accuracy depending on k number
    plt.figure(figsize=[16, 9])
    plt.plot(np.arange(1, 51), all_accuracies, label="Testing accuracy")
    plt.plot(np.arange(1, 51), all_train_accuracies, label="Training accuracy")
    plt.xticks(np.arange(1, 51))
    plt.legend()
    plt.title('KNN accuracy depending on number of neighbours')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    # plt.savefig('../img/generated/knn.png', format='png', bbox_inches='tight')


def plot_roc(x_test, y_test, classifier):
    """
    Plot ROC curve and calculate AUC

    :param x_test: testing data
    :param y_test: testing target array
    :param classifier: logistic regression classifier
    """

    y_pred_prob = classifier.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred_prob, pos_label='Normal')

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'.format(auc(fpr, tpr)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_multiclass(x_test, y_test, classifier, classes):
    """
    Plot multiclass ROC curve and calculate AUC for each class

    :param x_test: testing data
    :param y_test: testing target array
    :param classifier: logistic regression classifier
    :param classes: unique classes of target array
    """

    # predict confidence scores for each class
    y_score = classifier.decision_function(x_test)

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

    # plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    for idx, i in enumerate(classes):
        plt.plot(fpr[idx], tpr[idx], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[idx]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc="lower right")
    plt.show()


def print_classification_report(y_test, y_pred):
    """
    Print classification report for the classifiers prediction
    :param y_test: real y values
    :param y_pred: predicted y values
    """

    classes = ['Hernia', 'Normal', 'Spondylolisthesis']

    # printing confusion matrix
    confusion = confusion_matrix(y_test, y_pred, labels=classes)
    print('Confusion Matrix\n')
    print(confusion)

    # printing accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))

    sensitivity = recall_score(y_test, y_pred, labels=classes, average=None)
    specificity = specificity_score(confusion, classes)

    for i in range(len(sensitivity)):
        print("Sensitivity " + classes[i] + ": " + str(sensitivity[i]))
        print("Specificity " + classes[i] + ": " + str(specificity[i]) + "\n")

    # print('Micro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='micro')))
    # print('Micro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='micro')))
    # print('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='micro')))
    #
    # print('Macro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='macro')))
    # print('Macro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='macro')))
    # print('Macro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='macro')))
    #
    # print('Weighted Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='weighted')))
    # print('Weighted Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='weighted')))
    # print('Weighted F1-score: {:.4f}'.format(f1_score(y_test, y_pred, average='weighted')))
    #
    # # together
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, digits=4))


def specificity_score(confusion, classes):

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