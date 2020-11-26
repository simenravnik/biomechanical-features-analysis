import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# constants
RANDOM_STATE = 10


def load_data():
    """
    Reading dataset
    :return: a pair of pandas data frames
    """
    data_2c = pd.read_csv('../data/dataset-2C.csv')
    data_3c = pd.read_csv('../data/dataset-3C.csv')

    return data_2c, data_3c


def column_distribution(x):
    """
    Plotting distribution histogram on which is fitted kernel density estimate (KDE)
    """
    sns.distplot(x)
    plt.show()


def pairwise_relationship(X):
    """
    Plotting multiple pairwise bivariate distributions in a dataset
    """
    sns.pairplot(X)
    sns.pairplot(X, hue="class")
    plt.show()


def class_count_plot(data_2c, data_3c):
    """
    Plotting the count of dataset class for both dataset
    """

    # plotting dataset with normal/abnormal class
    sns.countplot(x="class", data=data_2c, palette="Set3")
    plt.show()

    # plotting dataset with hernia/spondylolisthesis/normal class
    sns.countplot(x="class", data=data_3c, palette="Set3")
    plt.show()


def univariate_analysis(data):
    """
    Statistics of each variable by itself

    :param data: all data
    """
    print('Data types:')
    print(data.dtypes)

    print('Shape:')
    print(data.shape)

    print('Describe:')
    print(data.describe())

    print('Missing values:')
    print(data.apply(lambda x: sum(x.isnull()), axis=0))

    # plotting class count
    class_count_plot(data_2c, data_3c)

    # plotting distribution for each column in dataset
    for column in data_2c.drop('class', axis=1):
        column_distribution(data_2c[column])


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


def hierarchical_clustering(data_3c):

    # convert pandas data frame to numpy array
    data_matrix = data_3c.to_numpy()

    # remove class column from the data
    data = data_matrix[:, 0:-1]
    class_column = data_3c.iloc[:, -1]

    plt.figure(figsize=(30, 40))
    plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
    plt.xlabel('distance', fontsize=12)

    # np array of target value
    labels_arr = np.array(list(class_column))

    # plotting dendrogram of hierarchical clustering with ward linkage
    shc.dendrogram(
        shc.linkage(data, method='ward'),
        labels=labels_arr,
        leaf_font_size=6,
        color_threshold=220,
        orientation='left'
    )
    plt.savefig('../img/generated/hierarchical_clustering.png', format='png', bbox_inches='tight')

    # hierarchical clustering for plotting scatter plot
    cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)
    plt.scatter(data[:, 1], data[:, 4], c=cluster.labels_, cmap='rainbow')
    plt.show()


def calculate_pca(data):
    """
    Calculates PCA for both two and three principal components along with plotting them.
    Plus visualise cumulative explained variance ration depending on number of components.

    :param data: all data from the dataset
    """

    # separating the features from the target value
    x = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # centering the data
    x = x - np.mean(x, axis=0)

    # performing two components pca
    pca_matrix_two_components = n_components_pca(2, data, x, ['PC1', 'PC2'])
    plot_pca(2, pca_matrix_two_components)

    # performing three components pca
    pca_matrix_three_components = n_components_pca(3, data, x, ['PC1', 'PC2', 'PC3'])
    for vertical_angle in range(0, 91, 45):
        for horizontal_angle in range(0, 91, 45):
            plot_pca(3, pca_matrix_three_components, horizontal_angle=horizontal_angle, vertical_angle=vertical_angle)

    # displaying cumulative explained variance ratio depending on number of components
    cumulative_explained_variance_ratio(x)


def n_components_pca(n, data, x, columns):
    """
    :param n: number of components
    :param data: all data from the dataset
    :param x: centered (standardized) data
    :param columns: list of components names (must be the size of n)
    :return: concatenated matrix of principal components and target value
    """
    # performing PCA
    pca = PCA(n_components=n)  # two components
    principal_components = pca.fit_transform(x)  # fitting the data

    # transforming the data matrix into pd data frame
    principal_df = pd.DataFrame(data=principal_components, columns=columns)

    # concatenating the target values to the principal component data frame
    final_df = pd.concat([principal_df, data['class']], axis=1)

    # printing explained variance ratio
    print(pca.explained_variance_ratio_)

    return final_df


def plot_pca(n, pca_matrix, horizontal_angle=None, vertical_angle=None):
    """
    :param n: number of components
    :param pca_matrix: concatenated matrix of principal components and target value
    :param horizontal_angle: is an optional parameter which defines horizontal angle of the figure
    :param vertical_angle: is an optional parameter which defines vertical angle of the figure
    """
    plt.figure(figsize=(8, 8))

    # adding figure metadata depending on number of components
    ax = None
    if n == 2:
        ax = plt.axes()
        ax.set_title('2 component PCA', fontsize=20)
    elif n == 3:
        ax = plt.axes(projection='3d')
        ax.set_title('3 component PCA', fontsize=20)
        ax.set_zlabel('PC3')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # defining colors for each or the target values
    targets = list(set(pca_matrix['class']))    # unique elements of class column
    colors = ['r', 'g', 'b']

    # for each of the target values ilocate indexes and
    # plotting them on scatter plot with the defined color
    for target, color in zip(targets, colors):
        idx_to_keep = pca_matrix['class'] == target

        if n == 2:
            ax.scatter(pca_matrix.loc[idx_to_keep, 'PC1'],
                       pca_matrix.loc[idx_to_keep, 'PC2'],
                       c=color, s=50)
        elif n == 3:
            ax.scatter(pca_matrix.loc[idx_to_keep, 'PC1'],
                       pca_matrix.loc[idx_to_keep, 'PC2'],
                       pca_matrix.loc[idx_to_keep, 'PC3'],
                       c=color, s=50)

    ax.legend(targets)
    ax.grid()

    # setting an angle of the figure
    if horizontal_angle is not None and vertical_angle is not None:
        ax.view_init(vertical_angle, horizontal_angle)

    plt.show()


def cumulative_explained_variance_ratio(x):
    """
    Plot explained variance depending on the number of principal components

    :param x: centered (standardized) data
    """
    pca = PCA().fit(x)

    cumulative_ratio = np.zeros(len(pca.explained_variance_ratio_) + 1)
    cumulative_ratio[1:] = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(14, 9))
    plt.plot(cumulative_ratio, label="Cumulative explained variance")
    plt.title('PCA explained variance ratio')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.legend(loc="lower right")
    plt.savefig('../img/generated/PCA_explained_variance_ratio.png', format='png', bbox_inches='tight')


def calculate_features_importance(data):
    """
    :param data: all data from the dataset
    """
    # separating the features from the target value
    x = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # calculating and plotting features importance
    plot_features_importance(data, x, target)

    # calculating features correlation and plotting heat map
    plot_correlation_matrix(data)


def plot_features_importance(data, x, y):
    """
    :param data: all data from the dataset
    :param x: separated features from the target value
    :param y: target value vector
    """
    # fitting a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset
    model = ExtraTreesClassifier()
    model.fit(x, y)

    # plotting chart of feature importance
    column_names = data.drop('class', axis=1).columns
    feature_importance = pd.Series(model.feature_importances_, index=column_names)
    feature_importance.plot(kind='barh')
    plt.show()


def plot_correlation_matrix(data):
    """
    :param data: all data from the dataset
    """
    plt.figure(figsize=(15, 10))
    sns.heatmap(data.corr(), annot=True)
    plt.show()


def predict_knn(data):
    """
    Training and predicting with KNN model

    :param data: all data from the dataset
    """
    # separating the features from the target value
    x_data = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # normalizing the data (standardizing the features)
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

    # splitting data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, target, random_state=RANDOM_STATE, test_size=0.2, shuffle=True)

    # calculating optimal k for KNN prediction model
    optimal_k = calculate_optimal_k(x_train, y_train)

    # defining prediction model with k nearest neighbours
    knn = KNeighborsClassifier(n_neighbors=optimal_k)

    # fitting k-fold train data into the model
    knn.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = cross_validation_score(x_train, y_train, KNeighborsClassifier(n_neighbors=optimal_k))
    print('Cross validation mean accuracy: ', cross_val_accuracy)

    # accuracy of KNN model
    accuracy = knn.score(x_test, y_test)

    print('Optimal k for KNN calculated: ', optimal_k)
    print('KNN prediction model accuracy: ', accuracy)

    # printing classification report
    y_pred = knn.predict(x_test)
    print_classification_report(y_test, y_pred)


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

    # plotting accuracy depending on k number
    plt.figure(figsize=[16, 9])
    plt.plot(np.arange(1, 51), all_accuracies, label="Testing accuracy")
    plt.plot(np.arange(1, 51), all_train_accuracies, label="Training accuracy")
    plt.xticks(np.arange(1, 51))
    plt.legend()
    plt.title('KNN accuracy depending on number of neighbours')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.savefig('../img/generated/knn.png', format='png', bbox_inches='tight')

    # transforming list into numpy array
    all_accuracies = np.array(all_accuracies)

    # the best k for KNN algorithm is the index of max mean accuracies
    optimal_k = np.argmax(all_accuracies) + 1      # +1 because we dont start with zero

    return optimal_k


def predict_log_regression(data):
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
    x_train, x_test, y_train, y_test = train_test_split(x, target, random_state=RANDOM_STATE, test_size=0.2, shuffle=True)

    classifier = LogisticRegression(multi_class='ovr')
    classifier.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = cross_validation_score(x_train, y_train, LogisticRegression(multi_class='ovr'))
    print('Cross validation mean accuracy: ', cross_val_accuracy)

    # printing accuracy of the model
    print(classifier.score(x_test, y_test))

    # plotting ROC and calculating AUC
    if len(classifier.classes_) == 2:
        plot_roc(x_test, y_test, classifier)    # two class roc plotting
    else:
        plot_roc_multiclass(x_test, y_test, classifier, classifier.classes_)    # multiclass ROC plotting

    # printing classification report
    y_pred = classifier.predict(x_test)
    print_classification_report(y_test, y_pred)


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

    binary_y_test = np.array(binary_y_test)     # convert to numpy array

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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def predict_decision_tree(data):
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
    x_train, x_test, y_train, y_test = train_test_split(x, target, random_state=RANDOM_STATE, test_size=0.2, shuffle=True)

    classifier = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    classifier.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = cross_validation_score(x_train, y_train, DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE))
    print('Cross validation mean accuracy: ', cross_val_accuracy)

    # printing accuracy of the model on test data
    print(classifier.score(x_test, y_test))

    # printing classification report
    y_pred = classifier.predict(x_test)
    print_classification_report(y_test, y_pred)


def predict_random_forest(data):
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
    x_train, x_test, y_train, y_test = train_test_split(x, target, random_state=RANDOM_STATE, test_size=0.2, shuffle=True)

    classifier = RandomForestClassifier(n_estimators=20, random_state=RANDOM_STATE)
    classifier.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = cross_validation_score(x_train, y_train, RandomForestClassifier(n_estimators=20, random_state=RANDOM_STATE))
    print('Cross validation mean accuracy: ', cross_val_accuracy)

    # printing accuracy of the model on test data
    print(classifier.score(x_test, y_test))

    # printing classification report
    y_pred = classifier.predict(x_test)
    print_classification_report(y_test, y_pred)


def predict_multilayer_perceptron(data):
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
    x_train, x_test, y_train, y_test = train_test_split(x, target, random_state=RANDOM_STATE, test_size=0.2, shuffle=True)

    classifier = MLPClassifier(solver='lbfgs', max_iter=300, random_state=RANDOM_STATE)
    classifier.fit(x_train, y_train)

    # calculating k-fold cross validation score
    cross_val_accuracy = cross_validation_score(x_train, y_train, MLPClassifier(solver='lbfgs', max_iter=300, random_state=RANDOM_STATE))
    print('Cross validation mean accuracy: ', cross_val_accuracy)

    # printing accuracy of the model on test data
    print(classifier.score(x_test, y_test))

    # printing classification report
    y_pred = classifier.predict(x_test)
    print_classification_report(y_test, y_pred)


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


def print_classification_report(y_test, y_pred):
    """
    Print classification report for the classifiers prediction
    :param y_test: real y values
    :param y_pred: predicted y values
    """

    # printing confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)

    # printing accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.4f}'.format(f1_score(y_test, y_pred, average='weighted')))

    # together
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    # load data
    data_2c, data_3c = load_data()

    # univariate analysis
    univariate_analysis(data_3c)

    # removing outliers from the dataset
    data_3c, _ = remove_outliers(data_3c)
    data_2c, _ = remove_outliers(data_2c)

    # plotting pairwise relationships
    pairwise_relationship(data_3c)

    # CLUSTERING
    hierarchical_clustering(data_3c)

    # dimensionality reduction
    calculate_pca(data_3c)

    # features importance and correlation
    calculate_features_importance(data_3c)

    # PREDICTION
    predict_knn(data_3c)                    # K-nearest neighbours prediction
    predict_log_regression(data_3c)         # logistic regression prediction
    predict_decision_tree(data_3c)          # decision tree prediction
    predict_random_forest(data_3c)          # random forest prediction
    predict_multilayer_perceptron(data_3c)  # neural network prediction
