import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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


def predict(data, classifier):
    """
    Main function for predicting.

    :param data: all data from the dataset
    :param classifier: wanted classifier for prediction
    """
    # Lists for storing accuracy measurements
    accuracy_arr = []
    cross_val_arr = []
    f1_score_arr = []

    # dictionary for storing sensitivities and specificities of Hernia and Spondylolisthesis
    diagnosis_dict = {
        "sensitivity_hernia_arr": [],
        "specificity_hernia_arr": [],
        "sensitivity_spondylolisthesis_arr": [],
        "specificity_spondylolisthesis_arr": []
    }

    # we run algorithm n times to get distribution of accuracies of certain classifier
    for i in range(100):
        # predicting with wanted classifier
        accuracy, cross_val_accuracy, f1, current_diagnosis = classifier(data)

        # appending calculated accuracy and cross validation accuracy to list of accuracies and cross val scores
        accuracy_arr.append(accuracy)
        cross_val_arr.append(cross_val_accuracy)
        f1_score_arr.append(f1)

        # appending calculated sensitivity and specificity to diagnosis dictionary
        diagnosis_dict["sensitivity_hernia_arr"].append(current_diagnosis["sensitivity_hernia"])
        diagnosis_dict["specificity_hernia_arr"].append(current_diagnosis["specificity_hernia"])
        diagnosis_dict["sensitivity_spondylolisthesis_arr"].append(current_diagnosis["sensitivity_spondylolisthesis"])
        diagnosis_dict["specificity_spondylolisthesis_arr"].append(current_diagnosis["specificity_spondylolisthesis"])

    # plotting distributions of accuracies and cross val scores
    plot_distribution(accuracy_arr)
    # plot_distribution(cross_val_arr)

    # average numbers
    print("Average accuracy = ", statistics.mean(accuracy_arr))
    print("Average cross validation accuracy = ", statistics.mean(cross_val_arr))
    print("F1 score (micro) = ", statistics.mean(f1_score_arr))

    # calculating confidence intervals for sensitivity and specificity
    diagnosis_conf_dict = helper.calculate_confidence_intervals(
        diagnosis_dict["sensitivity_hernia_arr"],
        diagnosis_dict["specificity_hernia_arr"],
        diagnosis_dict["sensitivity_spondylolisthesis_arr"],
        diagnosis_dict["specificity_spondylolisthesis_arr"]
    )

    print("\nHernia diagnosis sensitivity confidence interval: ", diagnosis_conf_dict["sensitivity_hernia_conf"])
    print("Hernia diagnosis specificity confidence interval: ", diagnosis_conf_dict["specificity_hernia_conf"])

    print("\nSpondylolisthesis diagnosis sensitivity confidence interval: ",
          diagnosis_conf_dict["sensitivity_spondylolisthesis_conf"])
    print("Spondylolisthesis diagnosis specificity confidence interval: ",
          diagnosis_conf_dict["specificity_spondylolisthesis_conf"])


def knn(data):
    """
    Training and predicting with KNN model

    :param data: all data from the dataset
    """
    # separating the features from the target value
    x_data = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # normalizing data
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

    # splitting data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, target, stratify=target, test_size=0.3, shuffle=True)

    # calculating optimal k for KNN prediction model
    optimal_k = helper.calculate_optimal_k(x_train, y_train)

    # defining prediction model with k nearest neighbours
    classifier = KNeighborsClassifier(n_neighbors=optimal_k)

    # fitting k-fold train data into the model
    classifier.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = helper.cross_validation_score(x_train, y_train, KNeighborsClassifier(n_neighbors=optimal_k))

    # accuracy of KNN model
    accuracy = classifier.score(x_test, y_test)

    # print('Optimal k for KNN calculated: ', optimal_k)
    # print('KNN prediction model accuracy: ', accuracy)

    # f1 score
    y_pred = classifier.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='micro')

    # calculating sensitivity and specificity of current result
    df = helper.get_classification_report(y_test, y_pred)
    # dictionary for storing current sensitivity and specificity results
    diagnosis = {
        "sensitivity_hernia": df['Hernia']['Sensitivity'],
        "specificity_hernia": df['Hernia']['Specificity'],
        "sensitivity_spondylolisthesis": df['Spondylolisthesis']['Sensitivity'],
        "specificity_spondylolisthesis": df['Spondylolisthesis']['Specificity']
    }

    return accuracy, cross_val_accuracy, f1, diagnosis


def decision_tree(data):
    """
    Training and predicting with decision tree

    :param data: all data from the dataset
    """
    # separating the features from the target value
    x_data = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # standardizing features
    x = StandardScaler().fit_transform(x_data)

    # splitting data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, target, stratify=target, test_size=0.3, shuffle=True)

    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = helper.cross_validation_score(x_train, y_train, DecisionTreeClassifier(max_depth=3))

    # accuracy for decision tree model
    accuracy = classifier.score(x_test, y_test)

    # f1 score
    y_pred = classifier.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='micro')

    # calculating sensitivity and specificity of current result
    df = helper.get_classification_report(y_test, y_pred)
    # dictionary for storing current sensitivity and specificity results
    diagnosis = {
        "sensitivity_hernia": df['Hernia']['Sensitivity'],
        "specificity_hernia": df['Hernia']['Specificity'],
        "sensitivity_spondylolisthesis": df['Spondylolisthesis']['Sensitivity'],
        "specificity_spondylolisthesis": df['Spondylolisthesis']['Specificity']
    }

    return accuracy, cross_val_accuracy, f1, diagnosis


def log_regression(data):
    """
    Training and predicting with logistic regression

    :param data: all data from the dataset
    """
    # separating the features from the target value
    x_data = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # standardizing features
    x = StandardScaler().fit_transform(x_data)

    # splitting data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, target, stratify=target, test_size=0.3, shuffle=True)

    classifier = LogisticRegression(multi_class='ovr')
    classifier.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = helper.cross_validation_score(x_train, y_train, LogisticRegression(multi_class='ovr'))

    # accuracy of logistic regression model
    accuracy = classifier.score(x_test, y_test)

    # plotting ROC and calculating AUC
    # if len(classifier.classes_) == 2:
    #     plot_roc(x_test, y_test, classifier)    # two class roc plotting
    # else:
    #     plot_roc_multiclass(x_test, y_test, classifier, classifier.classes_)    # multiclass ROC plotting

    # f1 score
    y_pred = classifier.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='micro')

    # calculating sensitivity and specificity of current result
    df = helper.get_classification_report(y_test, y_pred)
    # dictionary for storing current sensitivity and specificity results
    diagnosis = {
        "sensitivity_hernia": df['Hernia']['Sensitivity'],
        "specificity_hernia": df['Hernia']['Specificity'],
        "sensitivity_spondylolisthesis": df['Spondylolisthesis']['Sensitivity'],
        "specificity_spondylolisthesis": df['Spondylolisthesis']['Specificity']
    }

    return accuracy, cross_val_accuracy, f1, diagnosis


def random_forest(data):
    """
    Training and predicting with random forest

    :param data: all data from the dataset
    """
    # separating the features from the target value
    x_data = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # standardizing features
    x = StandardScaler().fit_transform(x_data)

    # splitting data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, target, stratify=target, test_size=0.3, shuffle=True)

    classifier = RandomForestClassifier(n_estimators=128)
    classifier.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = helper.cross_validation_score(x_train, y_train, RandomForestClassifier(n_estimators=128))

    # accuracy for random forest model
    accuracy = classifier.score(x_test, y_test)

    # f1 score
    y_pred = classifier.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='micro')

    # calculating sensitivity and specificity of current result
    df = helper.get_classification_report(y_test, y_pred)
    # dictionary for storing current sensitivity and specificity results
    diagnosis = {
        "sensitivity_hernia": df['Hernia']['Sensitivity'],
        "specificity_hernia": df['Hernia']['Specificity'],
        "sensitivity_spondylolisthesis": df['Spondylolisthesis']['Sensitivity'],
        "specificity_spondylolisthesis": df['Spondylolisthesis']['Specificity']
    }

    return accuracy, cross_val_accuracy, f1, diagnosis


def multilayer_perceptron(data):
    """
    Training and predicting with random forest

    :param data: all data from the dataset
    """
    # separating the features from the target value
    x_data = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # standardizing features
    x = StandardScaler().fit_transform(x_data)

    # splitting data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, target, stratify=target, test_size=0.3, shuffle=True)

    classifier = MLPClassifier(solver='lbfgs', max_iter=500)
    classifier.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = helper.cross_validation_score(x_train, y_train, MLPClassifier(solver='lbfgs', max_iter=500))

    # accuracy for neural network model
    accuracy = classifier.score(x_test, y_test)

    # f1 score
    y_pred = classifier.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='micro')

    # calculating sensitivity and specificity of current result
    df = helper.get_classification_report(y_test, y_pred)
    # dictionary for storing current sensitivity and specificity results
    diagnosis = {
        "sensitivity_hernia": df['Hernia']['Sensitivity'],
        "specificity_hernia": df['Hernia']['Specificity'],
        "sensitivity_spondylolisthesis": df['Spondylolisthesis']['Sensitivity'],
        "specificity_spondylolisthesis": df['Spondylolisthesis']['Specificity']
    }

    return accuracy, cross_val_accuracy, f1, diagnosis


if __name__ == '__main__':
    # load data
    data_2c, data_3c = load_data()

    # removing outliers from the dataset
    data, _ = remove_outliers(data_3c)

    # PREDICTION
    print("\nKNN:")
    predict(data, knn)                       # K-nearest neighbours prediction

    print("\nDecision tree:")
    predict(data, decision_tree)             # decision tree prediction

    print("\nLogistic regression:")
    predict(data, log_regression)            # logistic regression prediction

    print("\nRandom forest:")
    predict(data, random_forest)             # random forest prediction

    print("\nMultilayer perceptron:")
    predict(data, multilayer_perceptron)     # neural network prediction
