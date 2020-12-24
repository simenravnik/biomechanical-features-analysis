import json
from functools import reduce
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


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

    # np array of target value
    labels_arr = np.array(list(class_column))

    clusters = shc.linkage(data, method='ward')

    T = shc.to_tree(clusters, rd=False)
    d3Dendro = dict(children=[], name="Root1")
    add_node(T, d3Dendro)

    label_tree(d3Dendro["children"][0], list(class_column), "Black")
    # Output to JSON
    json.dump(d3Dendro, open("d3-dendrogram.json", "w"), sort_keys=True, indent=4)


# Create a nested dictionary from the ClusterNode's returned by SciPy
def add_node(node, parent):
    # First create the new node and append it to its parent's children
    newNode = dict(node_id=node.id, distance=node.dist, children=[])
    parent["children"].append(newNode)
    parent["color"] = "Black"

    # Recursively add the current node's children
    if node.left:
        add_node(node.left, newNode)
    if node.right:
        add_node(node.right, newNode)


colors = ["#ff71ce", "#01cdfe", "#05ffa1", "#b967ff", "Magenta"]


# Label each node with the names of each leaf in its subtree
def label_tree(n, id2name, color):
    # If the node is a leaf, then we have its name
    if len(n["children"]) == 0:
        leaf_name = id2name[n["node_id"]]
        n["name"] = leaf_name
        n["color"] = color

    # If not, flatten all the leaves in the node's subtree
    else:
        # add color
        new_color = color
        if n["distance"] < 200:
            if color != 'Black':
                n["color"] = color
            else:
                new_color = colors.pop(0)
        reduce(lambda ls, c: label_tree(c, id2name, new_color), n["children"], [])

    # Delete the node id since we don't need it anymore and
    # it makes for cleaner JSON
    del n["node_id"]

    return n


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
    targets = list(set(pca_matrix['class']))  # unique elements of class column
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


if __name__ == '__main__':
    # load data
    data_2c, data_3c = load_data()

    # univariate analysis
    univariate_analysis(data_3c)

    # removing outliers from the dataset
    # data_3c, outliers = remove_outliers(data_3c)
    data_2c, _ = remove_outliers(data_2c)

    # plotting pairwise relationships
    pairwise_relationship(data_3c)

    # CLUSTERING
    hierarchical_clustering(data_3c)

    # dimensionality reduction
    calculate_pca(data_3c)

    # features importance and correlation
    calculate_features_importance(data_3c)
