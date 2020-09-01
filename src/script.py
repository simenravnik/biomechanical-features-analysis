import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


def load_data():
    """
    Reading data sets
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
    Plotting the count of dataset class for both data sets
    """

    # plotting dataset with normal/abnormal class
    sns.countplot(x="class", data=data_2c, palette="Set3")
    plt.show()

    # plotting dataset with hernia/spondylolisthesis/normal class
    sns.countplot(x="class", data=data_3c, palette="Set3")
    plt.show()


def hierarchical_clustering(data_3c):

    # convert pandas data frame to numpy array
    data_matrix = data_3c.to_numpy()

    # remove class column from the data
    data = data_matrix[:, 0:-1]
    class_column = data_3c.iloc[:, -1]

    # plotting dendogram of hierarchical clustering with ward linkage
    shc.dendrogram(shc.linkage(data, method='ward'))
    plt.show()

    # hierarchical clustering for plotting scatter plot
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)
    plt.scatter(data[:, 1], data[:, 4], c=cluster.labels_, cmap='rainbow')
    plt.show()


def calculate_pca(data):
    """
    Calculates PCA for both two and three principal components along with plotting them.
    Plus visualise cumulative explained variance ration depending on number of components.

    :param data: all data from the data set
    """

    # separating the features from the target value
    x = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # centering the data (standardizing the features)
    x = StandardScaler().fit_transform(x)

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
    :param data: all data from the data set
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
        ax.set_zlabel('Principal Component 3')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

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

    plt.plot(cumulative_ratio)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


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


if __name__ == '__main__':

    # load data
    data_2c, data_3c = load_data()

    # plotting distribution for each column in dataset
    for column in data_2c.drop('class', axis=1):
        column_distribution(data_2c[column])

    # plotting pairwise relationships
    pairwise_relationship(data_3c)

    # plotting class count
    class_count_plot(data_2c, data_3c)

    hierarchical_clustering(data_3c)

    # dimensionality reduction
    calculate_pca(data_3c)

    # features importance and correlation
    calculate_features_importance(data_3c)

